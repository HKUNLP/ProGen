from datasets import load_metric, load_dataset, load_from_disk
import os
from .glue_processor import GLUEProcessor


class TCProcessor(GLUEProcessor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)

    def load_dataset(self):
        def _reverse_label(ex):
            ex['label'] = abs(ex['label']-1)
            return ex

        data_path = f'data/{self.task_name}'
        if os.path.exists(data_path):
            self.dataset = load_from_disk(data_path)
        else:
            if self.task_name == "sst-2":
                self.dataset = load_dataset('gpt3mix/sst2')
            else:
                self.dataset = load_dataset(self.task_name)
            self.dataset.save_to_disk(data_path)

        for name, subset in self.dataset.items():
            self.dataset[name] = subset.add_column('idx', list(range(len(subset))))

        # if no val set available, we reported metric on test set
        if self.validation_key not in self.dataset:
            self.validation_key = self.test_key

        if self.task_name in ['sst-2', 'elec']:
            # change to 0: neg, 1: pos
            self.dataset = self.dataset.map(_reverse_label, batched=False, load_from_cache_file=False)

        self.sentence1_key, self.sentence2_key = 'text', None
        self.encoded_dataset = self._encode_dataset(self.dataset)
        self.metric = load_metric("glue", "sst2")
        self.main_metric_name = "eval_accuracy"

    def _preprocess_function(self, examples):
        return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=512)

