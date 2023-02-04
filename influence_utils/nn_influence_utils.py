# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union, Optional, Tuple, Iterator, Any
import torch.nn.functional as F


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def convert_ids_to_string(
        tokenizer: PreTrainedTokenizer,
        ids: torch.LongTensor) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(tokens)


def get_loss_with_weight_decay(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        obj: str = "ce"
) -> float:
    model.train()
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    outputs = model(**inputs)
    # model outputs are always tuple in transformers (see doc)
    loss = outputs['loss']
    logits = outputs['logits']
    labels = inputs['labels']

    if obj == "rce" or obj == "sce":
        one_hot = labels.new_zeros(len(labels), 2, dtype=float).scatter_(1, labels.view(-1, 1), 1)
        one_hot = F.softmax(one_hot, dim=1)
        rce_loss = F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) - torch.log(one_hot)) - torch.mul(
                    F.softmax(logits, dim=1), F.log_softmax(logits, dim=1))
        rce_loss = rce_loss.sum(-1).mean()
        if obj == "sce":
            loss = rce_loss + 0.1 * loss
        else:
            loss = rce_loss

    if n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()

    # In PyTorch, weight-decay loss and gradients are calculated in
    # optimizers rather in nn.Module, so we have to manually specify
    # this for the loss here.
    if weight_decay is not None:
        no_decay = (
            weight_decay_ignores
            if weight_decay_ignores
               is not None else [])

        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay
        loss = loss + weight_decay_loss

    return loss


def compute_gradients(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Union[Dict[str, torch.Tensor], torch.utils.data.DataLoader],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        obj: str = 'ce'
) -> List[torch.FloatTensor]:
    if params_filter is None:
        params_filter = []

    model.zero_grad()
    if isinstance(inputs, torch.utils.data.DataLoader):
        grad = None
        for _inputs in inputs:
            _grad = compute_gradients(
                model=model,
                n_gpu=n_gpu,
                device=device,
                inputs=_inputs,
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores,
                obj=obj)
            if grad is None:
                grad = _grad
            else:
                grad = [a+b for a, b in zip(grad, _grad)]
        grad = [a / len(inputs) for a in grad]  # average over all instances, note here assume using mean loss
    else:
        loss = get_loss_with_weight_decay(
            device=device, n_gpu=n_gpu,
            model=model, inputs=inputs,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            obj=obj
        )

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=[
                param for name, param
                in model.named_parameters()
                if name not in params_filter],
            create_graph=True)

    return [a.detach() for a in grad]


def compute_hessian_vector_products(
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        vectors: torch.FloatTensor,
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]
) -> List[torch.FloatTensor]:
    if params_filter is None:
        params_filter = []

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        model=model, n_gpu=n_gpu,
        device=device, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        grad_outputs=vectors,
        only_inputs=True
    )
    return [a.detach() for a in grad_grad_tuple]


def compute_s_test(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Union[Dict[str, torch.Tensor], torch.utils.data.DataLoader],
        train_data_loaders: List[torch.utils.data.DataLoader],
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        damp: float,
        scale: float,
        s_test_iterations: int = 1,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        s_test_obj: str = "ce"
) -> List[torch.FloatTensor]:
    v = compute_gradients(
        model=model,
        n_gpu=n_gpu,
        device=device,
        inputs=test_inputs,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        obj=s_test_obj)
    if verbose is True:
        print("init v norm: ", v[0].norm().item())
    inverse_hvp = None
    for _ in range(s_test_iterations):
        # Technically, it's hv^-1
        last_estimate = list(v).copy()
        with tqdm(total=num_samples) as pbar:
            for data_loader in train_data_loaders:
                for i, inputs in enumerate(data_loader):
                    this_estimate = compute_hessian_vector_products(
                        model=model,
                        n_gpu=n_gpu,
                        device=device,
                        vectors=last_estimate,
                        inputs=inputs,
                        params_filter=params_filter,
                        weight_decay=weight_decay,
                        weight_decay_ignores=weight_decay_ignores)
                    # Recursively caclulate h_estimate
                    # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
                    with torch.no_grad():
                        new_estimate = [
                            a + (1 - damp) * b - c / scale
                            for a, b, c in zip(v, last_estimate, this_estimate)
                        ]

                    pbar.update(1)
                    if verbose is True:
                        new_estimate_norm = new_estimate[0].norm().item()
                        last_estimate_norm = last_estimate[0].norm().item()
                        estimate_norm_diff = new_estimate_norm - last_estimate_norm
                        pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")

                    last_estimate = new_estimate
                    if num_samples is not None and i > num_samples:
                        break

        if inverse_hvp is None:
            inverse_hvp = [X / scale for X in last_estimate]
        else:
            inverse_hvp = [
                pre + X / scale for pre, X in zip(inverse_hvp, last_estimate)
            ]

    # average across multiple runs
    inverse_hvp = [i / s_test_iterations for i in inverse_hvp]
    return inverse_hvp


def compute_influences(
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs: Union[Dict[str, torch.Tensor], torch.utils.data.DataLoader],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        s_test_obj: str = "ce",
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
        verbose: bool = True,
) -> Tuple[Dict[int, float], List[torch.FloatTensor]]:
    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = compute_s_test(
            n_gpu=n_gpu,
            device=device,
            model=model,
            test_inputs=test_inputs,
            train_data_loaders=[batch_train_data_loader],
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            damp=s_test_damp,
            scale=s_test_scale,
            s_test_iterations=s_test_iterations,
            num_samples=s_test_num_samples,
            verbose=verbose,
            s_test_obj=s_test_obj
        )

    influences = {}
    for index, train_inputs in enumerate(tqdm(instance_train_data_loader)):

        # Skip indices when a subset is specified to be included
        if (train_indices_to_include is not None) and (
                index not in train_indices_to_include):
            continue

        if (precomputed_grad_train_dict is not None and index not in precomputed_grad_train_dict) or \
                precomputed_grad_train_dict is None:
            grad_z = compute_gradients(
                n_gpu=n_gpu,
                device=device,
                model=model,
                inputs=train_inputs,
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores)
            if precomputed_grad_train_dict is not None:
                precomputed_grad_train_dict[index] = [i.cpu() for i in grad_z]
        else:
            grad_z = [i.to(s_test[0].device) for i in precomputed_grad_train_dict[index]]

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)]

        influences[index] = sum(influence).item()

    return influences, s_test
