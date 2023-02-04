# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains some utility functions.
"""

import json
import logging
import os
import random
import sys
from typing import List, Optional, Dict, Union, Tuple
from transformers import GPT2Tokenizer
import pandas as pd
from datasets import Dataset
from influence_utils import experiments
import numpy as np
import torch

PLACEHOLDER_C = "<C>"
PLACEHOLDER_X = "<X>"
PLACEHOLDER_Y = "<Y>"
PLACEHOLDER_EXAMPLE = "<E>"

C_KEY = 'C'
X_KEY = 'X'
Y_KEY = 'Y'
E_KEY = 'E'
PROMPT_KEY = 'Prompt'


def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_jsonl(entries: List[Dict], path: str):
    with open(path, 'w', encoding='utf8') as fh:
        for entry in entries:
            fh.write(f'{json.dumps(entry)}\n')


def read_jsonl(path: str) -> List[Dict]:
    pairs = []
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            pairs.append(json.loads(line))
    return pairs


class IncontextSampler():
    def __init__(self, in_context_num=0, same_y=False, mix_y=False, order_type=1, keep_mapping=True, same_c=False,
                 labels=None):
        self.same_c = same_c
        self.same_y = same_y
        self.mix_y = mix_y
        self.order_type = order_type
        self.in_context_num = in_context_num
        self.keep_mapping = keep_mapping
        self.labels = labels
        self.pool = None

    def update_pool(self, dataset: List[Dict]):
        pool = pd.DataFrame(dataset)
        idx2name = {i: n for i, n in enumerate(pool.columns)}

        def _to_dict(g):
            res = []
            for item in g.values.tolist():
                res.append({idx2name[i]: v for i, v in enumerate(item) if i in idx2name})
            return res
        group_by = []
        if self.same_c:
            group_by.append(C_KEY)
        if self.same_y or self.mix_y:
            group_by.append(Y_KEY)

        if len(group_by) > 0:
            self.pool = pool.groupby(group_by).apply(_to_dict).to_dict()
        else:
            self.pool = pool.to_dict('records')

    @property
    def avaliable(self):
        return self.pool is not None and self.in_context_num > 0

    @property
    def size(self):
        def _size(dict_or_list):
            if isinstance(dict_or_list, list):
                return len(dict_or_list)
            size = {}
            for k, v in dict_or_list.items():
                size[k] = _size(v)
            return size

        return 0 if self.pool is None else _size(self.pool)

    def _sample(self, x, pool, in_context_num) -> List[Dict]:
        """select in_context_num examples from pool"""
        if len(pool) <= in_context_num:
            examples = pool
            np.random.shuffle(examples)
        else:
            examples = np.random.choice(pool, size=in_context_num, replace=False).tolist()

        if not self.keep_mapping:
            # random select a label for each in-context example
            for ex in examples:
                ex[Y_KEY] = random.choice(self.labels)

        if x is not None:
            examples = [ex for ex in examples if ex[X_KEY] != x]
        return examples

    def sample(self, c=None, x=None, y=None, n=None) -> List[Dict]:
        in_context_num = n if n is not None else self.in_context_num

        pool = self.pool

        group_by = []
        if self.same_c:
            group_by.append(c)
        if self.same_y:
            group_by.append(y)

        if len(group_by) == 1 and group_by[0] in pool:
            pool = pool[group_by[0]]
        elif len(group_by) > 1 and tuple(group_by) in pool:
            pool = pool[tuple(group_by)]

        if self.mix_y:
            num_examples_per_label = in_context_num // len(self.labels)
            label2examples = {}
            for l in self.labels:
                label2examples[l] = self._sample(x=x, pool=self.pool[l], in_context_num=num_examples_per_label)

            examples = []
            if self.order_type == 1:
                # random order
                for k, v in label2examples.items():
                    examples += v
                np.random.shuffle(examples)
            elif self.order_type == 2:
                # label0, label1, label2 | label0, label1, label2 ...
                for i in range(max([len(i) for i in label2examples.values()])):
                    for k, v in label2examples.items():
                        if len(v) > i:
                            examples.append(v[i])
            elif self.order_type in [3, 4, 5]:
                # type 3: 4 neg, 4 pos, pos:
                # type 4: 4 pos, 4 neg, pos:
                # type 5: 4 neg, 3 pos, pos:
                assert y is not None
                for k, v in label2examples.items():
                    if k != y:
                        examples += v
                if self.order_type == 3:
                    examples += label2examples[y]
                elif self.order_type == 4:
                    examples = label2examples[y] + examples
                else:
                    examples += label2examples[y][:-1]
        else:
            examples = self._sample(x=x, pool=pool, in_context_num=in_context_num)

        return examples


def build_instruction(instruction: str, c: Optional[Union[str, int]] = None, x: Optional[str] = None,
                      y: Optional[str] = None, e: Optional[List[str]] = None, truncate: bool = False,
                      tokenizer: GPT2Tokenizer = None, max_len: int = 900) -> Union[str, Tuple[str, List[str]]]:
    output = instruction
    if isinstance(c, int):
        return output

    if c is not None:
        output = output.replace(PLACEHOLDER_C, c)

    if x is not None:
        output = output.replace(PLACEHOLDER_X, x)

    if y is not None:
        output = output.replace(PLACEHOLDER_Y, y)

    if e is not None:
        if truncate:
            if len(e) == 0:
                output = output.replace(PLACEHOLDER_EXAMPLE, "")
                return output, []
            else:
                keep_exs = []
                total_len = len(tokenizer.tokenize(output))
                for _ex in e:
                    _len = len(tokenizer.tokenize(_ex))
                    if _len + total_len <= max_len:
                        keep_exs.append(_ex)
                        total_len += _len
                    else:
                        break
                output = output.replace(PLACEHOLDER_EXAMPLE, '\n\n'.join(keep_exs) + '\n\n')
                return output, keep_exs
        elif len(e) == 0:
            output = output.replace(PLACEHOLDER_EXAMPLE, "")
            return output
        else:
            output = output.replace(PLACEHOLDER_EXAMPLE, '\n\n'.join(e) + '\n\n')
            return output

    # if any placeholder is not set yet, reset to ""
    output = output.replace(PLACEHOLDER_C, "").replace(PLACEHOLDER_X, "").\
        replace(PLACEHOLDER_Y, "").replace(PLACEHOLDER_EXAMPLE, "")
    return output


def random_select_n(dataset: Dataset, n: Optional[int] = None):
    idx = np.arange(0, len(dataset))
    if n is not None and len(dataset) > n:
        np.random.shuffle(idx)
        idx = idx[:n]
        dataset = dataset.select(idx)
    return dataset, idx


def list_to_hf_dataset(entries: List[Dict], label_key: str, sentence1_key: str,
                       sentence2_key: Optional[str]) -> Dataset:
    res = {sentence1_key: [], label_key: [], 'idx': []}
    if sentence2_key is not None:
        res.update({sentence2_key: []})
    for i, entry in enumerate(entries):
        res[label_key].append(entry[Y_KEY])
        res['idx'].append(i)
        if sentence2_key is not None:
            res[sentence1_key].append(entry[C_KEY])
            res[sentence2_key].append(entry[X_KEY])
        else:
            res[sentence1_key].append(entry[X_KEY])
    return Dataset.from_dict(res)


def hf_dataset_to_list(dataset: Dataset, label_key: str, sentence1_key: str, sentence2_key: Optional[str]) -> List[
    Dict]:
    entries = []
    for item in list(dataset):
        if sentence2_key is not None:
            entry = {C_KEY: item[sentence1_key], X_KEY: item[sentence2_key], Y_KEY: item[label_key]}
        else:
            entry = {X_KEY: item[sentence1_key], Y_KEY: item[label_key]}
        entries.append(entry)
    return entries


def list2set(dataset: List[Dict]):
    return set([json.dumps(i) for i in dataset])


def set2list(dataset: set):
    return [json.loads(i) for i in list(dataset)]


def cal_influence(model, train_dataset, val_dataset, tokenizer, s_test_obj="ce", weight_decay=None,
                  num_train_to_use=None, num_val_to_use=None):
    in_context_pool, idx = random_select_n(train_dataset, num_train_to_use)
    val_dataset, _ = random_select_n(val_dataset, num_val_to_use)

    outputs_collections = experiments.run_full_influence_functions_dataset(
        model=model,
        train_dataset=in_context_pool,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        s_test_obj=s_test_obj,
        weight_decay=weight_decay
    )
    if_scores = np.array(list(outputs_collections[-1]['influences'].values()))
    return if_scores, idx


def get_helpful_harmful_indices(helpful_to_harmful_labels, helpful_to_harmful_scores, helpful_to_harmful_idx,
                                helpful_ratio, harmful_ratio):
    is_helpful = helpful_to_harmful_scores < 0
    selected_helpful_idx = np.zeros_like(helpful_to_harmful_idx, dtype=bool)
    selected_harmful_idx = np.zeros_like(helpful_to_harmful_idx, dtype=bool)
    normal_idx = np.arange(0, len(helpful_to_harmful_idx))

    def _get_at_label(l):
        is_label_l = helpful_to_harmful_labels == l
        is_label_l_helpful = is_label_l & is_helpful
        is_label_l_harmful = is_label_l
        label_l_helpful_num = int(helpful_ratio) if helpful_ratio > 1 else int(helpful_ratio * is_label_l_helpful.sum())
        label_l_harmful_num = int(harmful_ratio) if harmful_ratio > 1 else int(harmful_ratio * is_label_l_harmful.sum())

        if label_l_helpful_num > 0:
            label_l_helpful_idx = normal_idx[is_label_l_helpful][:label_l_helpful_num]
        else:
            label_l_helpful_idx = []

        if label_l_harmful_num > 0:
            label_l_harmful_idx = normal_idx[is_label_l_harmful][-label_l_harmful_num:]
        else:
            label_l_harmful_idx = []

        return label_l_helpful_idx, label_l_harmful_idx

    labels = set(helpful_to_harmful_labels)
    for l in labels:
        label_l_helpful_idx, label_l_harmful_idx = _get_at_label(l)
        selected_helpful_idx[label_l_helpful_idx] = True
        selected_harmful_idx[label_l_harmful_idx] = True

    helpful_idx = helpful_to_harmful_idx[selected_helpful_idx]
    helpful_scores = helpful_to_harmful_scores[selected_helpful_idx]
    harmful_idx = helpful_to_harmful_idx[selected_harmful_idx]
    harmful_scores = helpful_to_harmful_scores[selected_harmful_idx]
    return helpful_idx, helpful_scores, harmful_idx, harmful_scores


def test_get_helpful_harmful_indices():
    helpful_to_harmful_labels = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    helpful_to_harmful_idx = np.array([0, 1, 6, 7, 2, 4, 5, 3])
    helpful_to_harmful_scores = np.array([-2, -1, -0.5, -0.1, 0, 0.5, 1, 2])
    helpful_ratio = 50
    harmful_ratio = 0.05
    print(get_helpful_harmful_indices(helpful_to_harmful_labels, helpful_to_harmful_scores,
                                      helpful_to_harmful_idx, helpful_ratio, harmful_ratio))


if __name__ == '__main__':
    test_get_helpful_harmful_indices()
