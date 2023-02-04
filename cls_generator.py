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

import string
from functools import partial
from typing import Any
import numpy as np
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
from tqdm import tqdm

from generation import ModelWrapper
from tasks import Processor
from utils import *


class DataGenerator:
    """
    This class represents a generative language model which can be used to generate datasets from instructions.
    """

    def __init__(self, output_dir: str, task_name: str, instructions: Dict[str, Any], model: ModelWrapper = None,
                 max_length: int = 40, processor: Processor = None, min_length: int = 1,
                 exec_type: str = "gx", limit: Optional[int] = None, in_context_type: str = 'none',
                 in_context_num: int = 0, same_y: bool = False, mix_y: bool = False, same_c: bool = False,
                 order_type: int = 1, keep_mapping: bool = False,
                 remove_harmful: bool = False, in_context_ratio: Optional[float] = 1, remove_ratio: Optional[float] = 0,
                 feedback_ratio: float = 0, **kwargs):
        self.output_dir = output_dir
        self.model = model
        self.task_name = task_name
        self.max_length = max_length
        self.min_length = min_length
        self.generate_params = kwargs
        self.exec_type = exec_type
        self.processor = processor
        self.limit = limit

        self.build_instructions(instructions)
        self.in_context_type = in_context_type
        self.in_context_sampler = IncontextSampler(in_context_num=in_context_num, same_y=same_y, mix_y=mix_y,
                                                   order_type=order_type, keep_mapping=keep_mapping,
                                                   same_c=self.processor.sentence2_key is not None and same_c,
                                                   labels=self.labels)
        self.init_in_context_sampler()

        self.in_context_ratio = in_context_ratio
        self.remove_ratio = remove_ratio
        self.remove_harmful = remove_harmful
        self.feedback_ratio = feedback_ratio

    def build_instructions(self, instructions):
        self.labels = [float(y) if self.task_name == "stsb" else int(y) for y in instructions.keys()]
        self.gen_x_instruction = {label: instructions[str(label)]['gen_x_instruction'] for label in self.labels}
        self.gen_c_instruction = {label: instructions[str(label)]['gen_c_instruction'] for label in self.labels}
        self.example_instruction = {label: instructions[str(label)]['example_instruction'] for label in self.labels}
        self.prompting_instruction = {label: instructions[str(label)]['prompting_instruction'] for label in self.labels}


    def init_in_context_sampler(self):
        """ init in_context_sampler with gold validation set"""
        if self.in_context_type == 'val':
            concat_examples = hf_dataset_to_list(dataset=self.processor.dataset[self.processor.validation_key],
                                                 label_key=self.processor.label_key,
                                                 sentence1_key=self.processor.sentence1_key,
                                                 sentence2_key=self.processor.sentence2_key)
            self.in_context_sampler.update_pool(concat_examples)

    def prompting_inference(self, dataset, batch_size: int = 16, calibrate: bool = False):
        sentence1_key = self.processor.sentence1_key
        sentence2_key = self.processor.sentence2_key
        label_key = self.processor.label_key

        if calibrate:
            prior_data = [build_instruction(self.prompting_instruction[y]) for y in self.labels]
            logging.info(f"get class prior with: {str(prior_data)}")
            prior_ce_loss, lens = self.model.evaluate(prior_data)

        def preprocess_function(example, y):
            if sentence2_key is None:
                example = build_instruction(self.prompting_instruction[y],
                                            x=example[sentence1_key])
            else:
                example = build_instruction(self.prompting_instruction[y],
                                            c=example[sentence1_key],
                                            x=example[sentence2_key])
            example = ' '.join(example.split()[:800])  # todo, hardcode
            return {"prompt": example}

        datasets = []
        for y in self.labels:
            datasets.append(dataset.map(partial(preprocess_function, y=y),
                                        load_from_cache_file=False,
                                        remove_columns=dataset.column_names))

        def lm_loss(dataset, label_idx):
            dataloader = DataLoader(dataset['prompt'], batch_size=batch_size)
            loss_list = []
            with torch.no_grad():
                for step, batch in enumerate(tqdm(dataloader)):
                    ce_loss, lens = self.model.evaluate(batch)
                    if calibrate:  # -log(p(y|x)/p(y))
                        avg_loss = (ce_loss - prior_ce_loss[label_idx]) / lens
                    else:
                        avg_loss = ce_loss / lens
                    loss_list += avg_loss.tolist()
            return loss_list

        lm_loss_list = np.array([lm_loss(dataset, i) for i, dataset in enumerate(datasets)])
        preds = lm_loss_list.argmin(axis=0)
        gold = np.array(dataset[label_key])
        expanded_gold = np.expand_dims(gold, axis=0)
        lm_loss_exp = np.exp(lm_loss_list)
        gold_probs = np.take_along_axis(lm_loss_exp, expanded_gold, axis=0) / lm_loss_exp.sum(axis=0)
        acc = (preds == gold).sum() / len(preds)
        logging.info("Zero-shot prompting accuracy is " + str(acc))
        return gold_probs

    def generate_dataset(self, input_texts: Optional[List[str]], num_entries_per_input: Optional[int] = None,
                         batch_size: int = 16, log_every: int = 10000) -> List[Dict]:
        generate_with_inputs = input_texts is not None

        if generate_with_inputs:
            num_instructions = batch_size // num_entries_per_input
        else:
            if self.limit is None:
                self.limit = num_entries_per_input
            input_texts = [None for _ in range(num_entries_per_input)]
            num_entries_per_input = 1
            num_instructions = batch_size

        dataset = set()
        log_count = 1
        max_val = -1
        max_val_n = -1
        log_every_max = log_every
        log_every = 100  # for bad prompts, it may be hard to generate log_every examples without in-context examples
        fixed_val_dataset = None
        self.log = True
        while True:
            sampler = BatchSampler(SequentialSampler(input_texts), batch_size=num_instructions, drop_last=False)
            for i, indices in enumerate(tqdm(sampler)):
                c_list = [input_texts[i] for i in indices]

                for y in self.labels:
                    outputs = self._generate_dataset_entries(c_list=c_list, y=y, num_samples=num_entries_per_input)

                    if isinstance(dataset, list):
                        dataset = list2set(dataset)
                    dataset.update(list2set(outputs))  # deduplicate

                # split 1000 examples as synthetic val set
                if self.exec_type == 'gx' and (len(dataset) >= 2000 and fixed_val_dataset is None):
                    dataset, fixed_val_dataset = train_test_split(set2list(dataset), test_size=1000, shuffle=True,
                                                                  random_state=42)
                    dataset = list2set(dataset)
                    logging.info(f"Split {len(fixed_val_dataset)} data as fixed val dataset")

                # feedback
                if self.exec_type == 'gx' and self.processor and len(dataset) >= log_count * log_every:
                    dataset = set2list(dataset)
                    res_dict = {}

                    # show cases
                    table = wandb.Table(data=[[ex[PROMPT_KEY], ex[C_KEY], ex[X_KEY], ex[Y_KEY]]
                                              for ex in dataset[:100]],
                                        columns=[PROMPT_KEY, C_KEY, X_KEY, Y_KEY])
                    res_dict.update({'#Train': len(dataset), 'table': table})
                    # res_dict.update({'#Train': len(dataset)})

                    # re-init model and fine-tune from scratch
                    self.processor.load_model()  # use the initial model

                    if fixed_val_dataset is None:
                        train_dataset, val_dataset = train_test_split(dataset, train_size=0.9, shuffle=True,
                                                                      random_state=42)
                    else:
                        train_dataset, val_dataset = dataset, fixed_val_dataset
                    encoded_hf_train_dataset, encoded_hf_val_dataset = self.processor.load_train_val(train_dataset,
                                                                                                     val_dataset)
                    self.processor.train(encoded_hf_train_dataset, encoded_hf_val_dataset)
                    logging.info(f"Test results using {len(dataset)} training data: ")

                    res_dict.update({'syn-val': self.processor.validate(
                        val_dataset=encoded_hf_val_dataset
                    )[0][self.processor.main_metric_name]})

                    # check the metric on validation dataset with new model
                    val_metric, val_loss = self.processor.validate()
                    if val_metric[self.processor.main_metric_name] > max_val:
                        max_val = val_metric[self.processor.main_metric_name]
                        max_val_n = log_count
                    res_dict.update({"val": val_metric, "max_val": max_val, "max_val_n": max_val_n})

                    log_count += 1
                    self.log = True
                    log_every = min(log_every_max, log_every * 2)

                    logging.info("Save to disk...")
                    dataset_path = os.path.join(self.output_dir, f'{self.task_name}-dataset.jsonl')
                    save_jsonl(dataset, dataset_path)

                    if self.in_context_type.startswith('syn'):
                        in_context_dataset = train_dataset
                        if not self.in_context_type.endswith('rand') and len(train_dataset) > 10000:
                            if_scores, sample_idx = cal_influence(
                                model=self.processor.model,
                                train_dataset=encoded_hf_train_dataset,
                                val_dataset=self.processor.encoded_dataset[self.processor.validation_key]
                                if "if_gold" in self.in_context_type else encoded_hf_val_dataset,
                                tokenizer=self.processor.tokenizer,
                                s_test_obj="ce" if "if_gold" in self.in_context_type else 'rce',
                                # s_test_obj="ce",
                                weight_decay=0.005,
                                num_train_to_use=10000
                            )

                            helpful_to_harmful_idx = if_scores.argsort()
                            ori_helpful_to_harmful_idx = sample_idx[helpful_to_harmful_idx]
                            ori_helpful_to_harmful_score = if_scores[helpful_to_harmful_idx]
                            ori_labels = np.array(encoded_hf_train_dataset["label"])[ori_helpful_to_harmful_idx]

                            helpful_indices, helpful_scores, harmful_indices, harmful_scores = \
                                get_helpful_harmful_indices(
                                    helpful_to_harmful_labels=ori_labels,
                                    helpful_to_harmful_scores=ori_helpful_to_harmful_score,
                                    helpful_to_harmful_idx=ori_helpful_to_harmful_idx,
                                    helpful_ratio=self.in_context_ratio, harmful_ratio=self.remove_ratio
                                )

                            logging.info(f"helpful num/harmful num (remove? {self.remove_harmful}): "
                                         f"{len(helpful_indices)}/{len(harmful_indices)}")
                            for i in range(min(3, len(helpful_indices))):
                                logging.info(f"top-{i + 1} helpful example: {train_dataset[helpful_indices[i]]}, "
                                             f"score: {helpful_scores[i]}")
                            res_dict['helpful'] = len(helpful_indices)
                            res_dict['harmful'] = len(harmful_indices)

                            if self.in_context_type.endswith('helpful'):
                                in_context_dataset = [train_dataset[i] for i in helpful_indices]
                            elif self.in_context_type.endswith('harmful'):
                                in_context_dataset = [train_dataset[i] for i in harmful_indices]
                            else:
                                raise ValueError(f"Unsupported in_context_type {self.in_context_type}")

                            # remove harmful data
                            if self.remove_harmful:
                                dataset = [ex for i, ex in enumerate(train_dataset) if i not in harmful_indices]
                                logging.info(f"remove {len(harmful_indices)} from dataset, now size {len(dataset)}")

                        # self.in_context_sampler.update_full_pool(np.random.choice(dataset, 50, replace=False))
                        if len(in_context_dataset) > 0:
                            self.in_context_sampler.update_pool(in_context_dataset)
                            res_dict['pool_size'] = {str(k): v for k, v in self.in_context_sampler.size.items()} \
                                if isinstance(self.in_context_sampler.size, dict) else self.in_context_sampler.size

                    logging.info(res_dict)
                    wandb.log(res_dict)

                logging.info(f"Current dataset size: {len(dataset)}")
                if len(dataset) >= self.limit:
                    break

            if len(dataset) >= self.limit:
                break

        if isinstance(dataset, set):
            dataset = set2list(dataset)
        return dataset

    def _generate_dataset_entries(self, c_list: Union[List[str], List[None]], y: Any, num_samples: int) \
            -> List[Dict]:
        # filling in example instruction
        if self.processor and self.in_context_sampler.avaliable:
            # choose to use in-context example or not
            if random.random() > self.feedback_ratio:  # without in-context examples
                raw_in_context_examples = [[] for _ in c_list]
                in_context_examples = [[] for _ in c_list]
            else:
                raw_in_context_examples = [self.in_context_sampler.sample(y=y, c=c) for c in c_list]
                in_context_examples = [
                    [build_instruction(instruction=self.example_instruction[ex[Y_KEY]],
                                       c=ex[C_KEY] if C_KEY in ex else None,
                                       x=ex[X_KEY])
                     for ex in raw]
                    for raw in raw_in_context_examples]
        else:
            raw_in_context_examples = [[] for _ in c_list]
            in_context_examples = [[] for _ in c_list]

        # building full instruction by concatenating the example instruction (if applied)
        gen_instruction = self.gen_x_instruction if self.exec_type == 'gx' else self.gen_c_instruction
        instructions = [build_instruction(gen_instruction[y], c=c, e=example)
                        for c, example in zip(c_list, in_context_examples)]

        model_outputs = self.model.generate(
            input_texts=instructions,
            num_samples=num_samples,
            max_length=self.max_length,
            **self.generate_params
        )
        logging.info(f"Prompt: {instructions[0]}\n\nOutput: {model_outputs[0]}\n\n")

        outputs = []
        for i, (c, instruction, raw_in_context_example) in enumerate(
                zip(c_list, instructions, raw_in_context_examples)):
            if len(raw_in_context_example) > 0:
                example_x = [ex[X_KEY] for ex in raw_in_context_example]
            else:
                example_x = None
            for j in range(num_samples):
                output = process_output(c=c, output_text=model_outputs[i * num_samples + j],
                                        y=y, min_length=self.min_length, task_name=self.task_name,
                                        example_x=example_x, prompt=instruction)
                if output is not None:
                    outputs.append(output)

        if self.log:
            logging.info("=================Sample Prompts================")
            [logging.info(f'Prompt: {o[PROMPT_KEY]}\nOutput: {o[X_KEY]}') for o in outputs[:5]]
            logging.info('Filtered %d/%d examples' % (len(model_outputs) - len(outputs), len(model_outputs)))
            self.log = False
        return outputs


def process_output(c: Union[str, None], output_text: str, y: Any, min_length: int, task_name: str,
                   example_x: Optional[List[str]], prompt: str) -> Optional[Dict]:
    if task_name == "qnli":
        if '?' in output_text:
            output_text = output_text.split('?')[0] + "?"
        else:
            return None
    elif '"' in output_text:
        output_text = output_text.split('"')[0]
    elif '\n' in output_text:
        output_text = output_text.split('\n')[0]
    elif '.' in output_text:
        sentences = output_text.split('.')
        output_text = '.'.join(sentences[:-1]) + '.'
    else:
        return None

    if example_x:  # ignore the highly overlapped examples
        no_punc_table = str.maketrans('', '', string.punctuation)
        vocabs = set([i.strip() for i in output_text.lower().translate(no_punc_table).strip().split()])
        for x in example_x:
            x_vocabs = set([i.strip() for i in x.lower().translate(no_punc_table).strip().split()])
            overlap = len(vocabs.intersection(x_vocabs))
            if overlap / len(x_vocabs) >= 0.9:
                return None

    if len(output_text.strip().split(' ')) >= min_length:
        if c is not None:
            x = output_text
            if c == x:
                return None
        else:
            c = output_text
            x = None
        return {C_KEY: c, X_KEY: x, Y_KEY: y, PROMPT_KEY: prompt}
    return None
