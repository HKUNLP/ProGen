# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import List, Dict, Optional, Union

import torch
from datasets import Dataset
from tqdm import tqdm

from influence_utils import nn_influence_utils, faiss_utils, misc_utils
from transformers import default_data_collator


def compute_influences_single(
        model: torch.nn.Module,
        test_inputs: Union[Dict[str, torch.Tensor], Dataset],
        train_dataset: Dataset,
        batch_size: int = 1,
        data_collator=default_data_collator,
        s_test_damp: float = 5e-3,
        s_test_scale: float = 1e4,
        s_test_num_samples: int = 1000,
        s_test_iterations: int = 1,
        s_test_obj: str = "ce",
        k: int = None,
        faiss_index: faiss_utils.FAISSIndex = None,
        faiss_index_use_mean_features_as_query: bool = False,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        weight_decay: Optional[float] = None
) -> [Dict[int, float], List[List[torch.FloatTensor]]]:
    """calculate influence score for all train instance over single test input, when test_input is
        a dataset, then the influence score is on whole val dataset."""
    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
                               "bias",
                               "LayerNorm.weight"] + [
                               n for n, p in model.named_parameters()
                               if not p.requires_grad]

    if faiss_index is not None:
        features = misc_utils.compute_BERT_CLS_feature(model, test_inputs)
        features = features.cpu().detach().numpy()

        if faiss_index_use_mean_features_as_query is True:
            # We use the mean embedding as the final query here
            features = features.mean(axis=0, keepdims=True)

        KNN_distances, KNN_indices = faiss_index.search(
            k=k, queries=features)
    else:
        KNN_indices = None

    batch_train_data_loader = misc_utils.get_dataloader(
        train_dataset,
        batch_size=batch_size,
        random=True,
        data_collator=data_collator
    )

    instance_train_data_loader = misc_utils.get_dataloader(
        train_dataset,
        batch_size=1,
        random=False)

    if isinstance(test_inputs, Dataset):
        test_inputs = misc_utils.get_dataloader(
            test_inputs,
            batch_size=128,
            random=False,
            data_collator=data_collator)

    with torch.backends.cudnn.flags(enabled=False):
        influences, s_test = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=model.device,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            model=model,
            test_inputs=test_inputs,
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            s_test_iterations=s_test_iterations,
            s_test_obj=s_test_obj,
            train_indices_to_include=KNN_indices,
            precomputed_s_test=precomputed_s_test,
            precomputed_grad_train_dict=precomputed_grad_train_dict)

    return influences, s_test


def compute_influences_multiple(
        model: torch.nn.Module,
        val_dataset: Dataset,
        train_dataset: Dataset,
        batch_size: int = 1,
        data_collator=default_data_collator,
        s_test_damp: float = 5e-3,
        s_test_scale: float = 1e4,
        s_test_num_samples: int = 1000,
        k: int = None,
        faiss_index: faiss_utils.FAISSIndex = None,
        faiss_index_use_mean_features_as_query: bool = False,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        weight_decay: Optional[float] = None
) -> [Dict[int, float], List[List[torch.FloatTensor]]]:
    """calculate influence score for each train-val pair"""
    eval_instance_data_loader = misc_utils.get_dataloader(dataset=val_dataset,
                                                          batch_size=1,
                                                          random=False)
    output_collection = {}
    for test_index, val_input in enumerate(tqdm(eval_instance_data_loader)):
        precomputed_s_test = precomputed_s_test_dict[test_index] \
            if (precomputed_s_test_dict is not None and test_index in precomputed_s_test_dict) else None

        influences, s_test = compute_influences_single(
            k=k,
            faiss_index=faiss_index,
            model=model,
            test_inputs=val_input,
            train_dataset=train_dataset,
            batch_size=batch_size,
            data_collator=data_collator,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            device_ids=device_ids,
            weight_decay=weight_decay,
            precomputed_s_test=precomputed_s_test,
            precomputed_grad_train_dict=precomputed_grad_train_dict,
            faiss_index_use_mean_features_as_query=faiss_index_use_mean_features_as_query
        )

        if precomputed_s_test_dict is not None:
            precomputed_s_test_dict[test_index] = s_test

        outputs = {
            "influences": influences
        }
        output_collection[test_index] = outputs
    return output_collection
