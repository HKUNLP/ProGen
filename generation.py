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
This module contains various classes and functions required for text generation with self-debiasing.
"""
from typing import List, Tuple
from model_util import load_tokenizer_model
import torch
import numpy as np
import os

class ModelWrapper():

    def __init__(self, model_name: str = "gpt2-xl", pad_trunc_right: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        """
        super().__init__()
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer, self._model = load_tokenizer_model(model_name, pad_trunc_right)

    def evaluate(self, input_texts: List[str]) -> Tuple:
        URL = os.getenv("OPT_LOCAL_URL")
        if self.model_name == 'opt175b':
            import requests
            import json
            headers = {
                "Content-Type": "application/json; charset=UTF-8"
            }
            pyload = {"prompt": input_texts, "max_tokens": 0, "echo": True}
            response = json.loads(requests.post(URL, data=json.dumps(pyload), headers=headers).text)
            lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
            ce_loss = np.array([-sum(r['logprobs']['token_logprobs']) for r in response['choices']])
        else:
            inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=self._tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
                shift_labels.size())
            ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
            lens = (inputs["input_ids"] != self._tokenizer.pad_token_id).sum(-1).cpu().numpy()
        return ce_loss, lens

    def generate(self, input_texts: List[str],  num_samples: int = 1, max_length: int = None,
                 **kwargs) -> List[str]:
        if self.model_name == 'opt175b':
            import requests
            import json
            headers = {
                "Content-Type": "application/json; charset=UTF-8"
            }
            full_input_texts = []
            for text in input_texts:
                full_input_texts.extend([text]*num_samples)

            pyload = {"prompt": full_input_texts, "max_tokens": max_length, "top_p": kwargs['top_p'],
                      "temperature": kwargs["temperature"]}
            response = json.loads(requests.post(URL, data=json.dumps(pyload), headers=headers).text)
            # print('\n'.join([r['text'] for r in response['choices']]))
            return [r['text'] for r in response['choices']]
        else:
            inputs = self._tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
            batch_size, seq_len = inputs['input_ids'].shape

            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            if max_length is not None:
                max_length = min(self._model.config.max_position_embeddings, max_length + seq_len)

            output_ids = self._model.generate(**inputs, min_length=max_length, max_length=max_length,
                                              num_return_sequences=num_samples, **kwargs)
            # note here we assume padding is performed on the left
            output_ids = output_ids[:, seq_len:]
            return self._tokenizer.batch_decode(output_ids)
