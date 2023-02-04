# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import torch
import numpy as np
# from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from typing import Tuple, Optional, Union, Any, Dict, List, Callable
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    DataCollator,
    default_data_collator)
from datasets import Dataset
from influence_utils import model_utils


def sort_dict_keys_by_vals(d: Dict[int, float]) -> List[int]:
    sorted_items = sorted(list(d.items()),
                          key=lambda pair: pair[1])
    return [pair[0] for pair in sorted_items]


def sort_dict_keys_by_vals_with_conditions(
        d: Dict[int, float],
        condition_func: Callable[[Tuple[int, float]], bool]
) -> List[int]:

    sorted_items = sorted(list(d.items()),
                          key=lambda pair: pair[1])
    return [pair[0] for pair in sorted_items
            if condition_func(pair)]


def get_helpful_harmful_indices_from_influences_dict(
        d: Dict[int, float],
        n: Optional[int] = None,
) -> Tuple[List[int], List[int]]:

    helpful_indices = sort_dict_keys_by_vals_with_conditions(
        d, condition_func=lambda k_v: k_v[1] < 0.0)
    harmful_indices = sort_dict_keys_by_vals_with_conditions(
        d, condition_func=lambda k_v: k_v[1] > 0.0)[::-1]

    if n is not None:
        if len(helpful_indices) < n:
            raise ValueError(
                f"`helpful_indices` have only "
                f"{len(helpful_indices)} elememts "
                f"whereas {n} is needed")

        if len(harmful_indices) < n:
            raise ValueError(
                f"`harmful_indices` have only "
                f"{len(harmful_indices)} elememts "
                f"whereas {n} is needed")

        helpful_indices = helpful_indices[:n]
        harmful_indices = harmful_indices[:n]

    return helpful_indices, harmful_indices


def compute_BERT_CLS_feature(
    model,
    inputs
) -> torch.FloatTensor:
    if model.training is True:
        raise ValueError
    device = model.device
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    inputs = {k: v for k, v in inputs.items() if k != 'labels'}
    outputs = model.base_model(**inputs)
    features = outputs['last_hidden_state'][:, 0].contiguous()
    return features


def predict(model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:

    has_labels = any(
        inputs.get(k) is not None for k in
        ["labels", "lm_labels", "masked_lm_labels"])

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    step_eval_loss = None
    with torch.no_grad():
        outputs = model(**inputs)
        if has_labels:
            step_eval_loss, logits = outputs[:2]
        else:
            logits = outputs[0]

    preds = logits.detach()
    preds = preds.cpu().numpy()
    if inputs.get("labels") is not None:
        label_ids = inputs["labels"].detach()
        label_ids = label_ids.cpu().numpy()
    else:
        label_ids = None

    if step_eval_loss is not None:
        step_eval_loss = step_eval_loss.mean().item()

    return preds, label_ids, step_eval_loss


def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   random: bool = False,
                   data_collator: Optional[DataCollator] = None
                   ) -> DataLoader:
    if data_collator is None:
        data_collator = default_data_collator

    if random is True:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    return data_loader


def remove_file_if_exists(file_name: str) -> None:
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print("The file does not exist")


def is_prediction_correct(
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]]) -> bool:

    preds, label_ids, step_eval_loss = predict(
        model=model,
        inputs=inputs)

    if preds.shape[0] != 1:
        raise ValueError("This function only works on instances.")

    return bool((preds.argmax(axis=-1) == label_ids).all())
