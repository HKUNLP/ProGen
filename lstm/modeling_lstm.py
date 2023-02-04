#!/usr/bin/python3
# -*- coding: utf-8 -*-
from transformers import PreTrainedModel, PretrainedConfig, BertTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import math


class LSTMConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=100,
        n_layers=1,
        hidden_dim=300,
        embed_dropout=0.1,
        lstm_dropout=0,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed_dropout = embed_dropout
        self.lstm_dropout = lstm_dropout
        self.seq_classif_dropout = seq_classif_dropout
        super().__init__(**kwargs, pad_token_id=pad_token_id)


class LSTMTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" DistilBERT tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DistilBertTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    model_input_names = ["input_ids", "attention_mask"]


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        # output, (ht, ct) = self.bilstm(x, )
        # sent_emb = ht[-2:].permute(1, 0, 2).reshape(len(lens), -1)
        # return output, sent_emb

        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]

        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens.cpu(), batch_first=True)
        packed_output, (ht, ct) = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]

        sent_emb = ht[-2:].permute(1, 0, 2).reshape(len(lens), -1)
        sent_emb = sent_emb[recover_index]
        return recover_output, sent_emb


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.embed_dropout)
        nn.init.uniform_(self.word_embeddings.weight.data,
                         a=-math.sqrt(3 / self.word_embeddings.weight.data.size(1)),
                         b=math.sqrt(3 / self.word_embeddings.weight.data.size(1)))

    def forward(self, input_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class LSTMModel(PreTrainedModel):
    config_class = LSTMConfig
    base_model_prefix = "lstm"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pad_id = config.pad_token_id

        self.embed = Embeddings(config)
        self.bilstm = BiLSTM(config.embed_dim, config.hidden_dim, config.lstm_dropout, config.n_layers)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output = self.embed(input_ids)
        lstm_output, hc = self.bilstm(sequence_output, attention_mask.sum(-1))
        logits = self.classifier(self.dropout(hc))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


if __name__ == '__main__':
    tokenizer = LSTMTokenizerFast.from_pretrained("lstm")
    config = LSTMConfig.from_pretrained("lstm", num_labels=3)
    model = LSTMModel(config)
    model.save_pretrained("tmp_lstm")
    model = LSTMModel.from_pretrained("tmp_lstm")