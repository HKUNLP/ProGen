#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric, Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, IntervalStrategy, EarlyStoppingCallback, Trainer, AutoModelForSequenceClassification

from tasks.base_processor import Processor
from transformers.utils import logging
from utils import list_to_hf_dataset
from lstm.modeling_lstm import *


logger = logging.get_logger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("sentence", "question"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}


class GLUEProcessor(Processor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)
        self.train_key = "train"
        self.validation_key = "validation"
        self.test_key = "test"
        self.label_key = 'label'

        self.load_model()
        self.load_dataset()

    def load_model(self):
        self.num_labels = 3 if self.task_name.startswith("mnli") else 1 if self.task_name == "stsb" else 2
        self.is_regression = True if self.task_name == 'stsb' else False
        if self.model_name == 'lstm':
            self.tokenizer = LSTMTokenizerFast.from_pretrained("lstm")
            config = LSTMConfig.from_pretrained("lstm", num_labels=self.num_labels)
            self.model = LSTMModel.from_pretrained(self.model_ckpt) if self.model_ckpt is not None else LSTMModel(config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
            self.model = AutoModelForSequenceClassification.from_pretrained(load_name, num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def load_dataset(self):
        if self.task_name == 'mnli_mismatched':
            self.validation_key = "validation_mismatched"
            self.test_key = "test_mismatched"
            self.task_name = 'mnli'
        elif self.task_name == 'mnli_matched':
            self.validation_key = "validation_matched"
            self.test_key = "test_matched"
            self.task_name = 'mnli'

        data_path = f'data/{self.task_name}'
        if os.path.exists(data_path):
            self.dataset = load_from_disk(data_path)
        else:
            self.dataset = load_dataset('glue', self.task_name)
            self.dataset.save_to_disk(data_path)

        self.sentence1_key, self.sentence2_key = task_to_keys[self.task_name]
        self.encoded_dataset = self._encode_dataset(self.dataset)
        # no label for test set on GLUE tasks
        self.encoded_dataset[self.test_key] = self.encoded_dataset[self.test_key].remove_columns("label")
        self.metric = load_metric("glue", self.task_name)
        self.main_metric_name = "eval_spearman" if self.task_name == "stsb" \
            else "eval_matthews_correlation" if self.task_name == "cola" else "eval_accuracy"

    def _encode_dataset(self, dataset: Union[Dataset, DatasetDict]):
        remove_columns = [col for col in self.dataset[self.train_key].column_names if col != 'label']
        encoded_dataset = dataset.map(self._preprocess_function, batched=True,
                                      load_from_cache_file=False,
                                      remove_columns=remove_columns)
        return encoded_dataset

    def _preprocess_function(self, examples):
        if self.sentence2_key is None:
            return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=128)
        return self.tokenizer(examples[self.sentence1_key], examples[self.sentence2_key], truncation=True,
                              max_length=128)

    def _compute_metrics(self, eval_pred, metric_key_prefix='eval'):
        predictions, labels = eval_pred
        if self.is_regression:
            predictions = predictions[:, 0]
        else:
            predictions = np.argmax(predictions, axis=1)
        metrics = self.metric.compute(predictions=predictions, references=labels)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return metrics

    def load_train_val(self, train_dataset, val_dataset):
        def _load_hf_encoded_dataset(self, dataset):
            if isinstance(dataset, Dataset):
                if 'input_ids' not in dataset.column_names:
                    dataset = self._encode_dataset(dataset)
            elif isinstance(dataset, list):
                dataset = list_to_hf_dataset(dataset, self.label_key, self.sentence1_key, self.sentence2_key)
                if 'input_ids' not in dataset.column_names:
                    dataset = self._encode_dataset(dataset)
            else:
                raise RuntimeError()

            return dataset

        train_dataset = _load_hf_encoded_dataset(self, train_dataset)
        val_dataset = _load_hf_encoded_dataset(self, val_dataset)
        return train_dataset, val_dataset

    def train(self, train_dataset=None, val_dataset=None, val_examples=None, train=True):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.train_args['num_epochs'],
            per_device_train_batch_size=self.train_args['train_batch_size'],
            per_device_eval_batch_size=self.train_args['train_batch_size'],
            learning_rate=self.train_args['learning_rate'],
            weight_decay=0.005,
            evaluation_strategy=IntervalStrategy.EPOCH,
            metric_for_best_model=self.main_metric_name,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=1,
            load_best_model_at_end=True
        )
        self.trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics, data_collator=self.data_collator
        )
        if train:
            self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))
            self.trainer.train()

    def validate(self, val_dataset=None):
        if val_dataset is None:  # by default we use the standard validation set
            encoded_val_dataset = self.encoded_dataset[self.validation_key]
        elif isinstance(val_dataset, Dataset):
            if 'input_ids' not in val_dataset.column_names:
                encoded_val_dataset = self._encode_dataset(val_dataset)
            else:
                encoded_val_dataset = val_dataset
        else:
            raise RuntimeError()

        batch_size = self.train_args['train_batch_size']
        eval_dataloader = DataLoader(encoded_val_dataset,
                                     batch_size=batch_size,
                                     collate_fn=self.data_collator)

        self.model.eval()
        all_entropies = []
        loss = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits if self.is_regression else outputs.logits.argmax(dim=-1)
                loss += outputs.loss.item()
                probs = outputs.logits.softmax(dim=-1)
                entropy = (-probs.log()*probs).sum(-1)
                all_entropies += entropy.cpu().tolist()
                self.metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )

        eval_metric = self.metric.compute()
        # Prefix all keys with eval + '_'
        for key in list(eval_metric.keys()):
            if not key.startswith("eval_"):
                eval_metric[f"eval_{key}"] = eval_metric.pop(key)

        return eval_metric, loss/len(eval_dataloader)

    def predict(self, output_texts, label):
        x1_list = []
        x2_list = []
        labels = []
        for output_text in output_texts:
            segments = output_text.split('"', 3)  # split into 4 parts
            x1_list.append(segments[1])
            x2_list.append(segments[3])
            labels.append(int(label))
        if self.sentence2_key is None:
            inputs = self.tokenizer(x2_list, return_tensors="pt", truncation=False, padding=True).to(
                self.model.device)
        else:
            inputs = self.tokenizer(x1_list, x2_list, return_tensors="pt", truncation=False, padding=True).to(
                self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        labels = torch.tensor(labels, dtype=torch.long).to(logits.device)

        if self.is_regression:
            losses = nn.MSELoss(reduce=False)(logits, labels)
            return_probs = logits
        else:
            losses = nn.CrossEntropyLoss(reduce=False)(logits, labels)
            probs = logits.softmax(-1)

            # 1. entropy
            # entropy = (-probs.log()*probs).sum(-1)
            # return_probs = entropy

            # 2. other max prob
            probs[torch.arange(labels.size(0)).to(labels.device), labels] = 0
            other_max_probs = probs.max(-1)[0]
            return_probs = other_max_probs
        return losses, return_probs
