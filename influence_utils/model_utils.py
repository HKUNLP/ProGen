# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from transformers import DistilBertForSequenceClassification


def freeze_BERT_parameters(model: DistilBertForSequenceClassification, verbose: bool = True) -> None:
    # https://github.com/huggingface/transformers/issues/400
    if not isinstance(model, DistilBertForSequenceClassification):
        raise TypeError

    # Table 3 in https://arxiv.org/pdf/1911.03090.pdf
    params_to_freeze = [
        "distilbert.embeddings.",
        "distilbert.transformer.layer.0.",
        "distilbert.transformer.layer.1.",
        "distilbert.transformer.layer.2.",
        "distilbert.transformer.layer.3.",
    ]
    for name, param in model.named_parameters():
        # if "classifier" not in name:  # classifier layer
        #     param.requires_grad = False

        if any(pfreeze in name for pfreeze in params_to_freeze):
            param.requires_grad = False

    if verbose is True:
        num_trainable_params = sum([
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad])
        trainable_param_names = [
            n for n, p in model.named_parameters()
            if p.requires_grad]
        print(f"Params Trainable: {num_trainable_params}\n\t" +
              f"\n\t".join(trainable_param_names))
