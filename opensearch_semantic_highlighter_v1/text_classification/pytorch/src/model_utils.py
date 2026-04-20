# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper module defining the custom sentence-level BERT tagger architecture and
input-preparation utilities for OpenSearch Semantic Highlighter v1.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertTaggerForSentenceExtractionWithBackoff(BertPreTrainedModel):
    """Sentence-level BERT classifier with a confidence-backoff rule."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.dropout(outputs[0])

        def _get_agg_output(ids, seq_out):
            max_sentences = torch.max(ids) + 1
            d_model = seq_out.size(-1)

            agg_out, global_offsets, num_sents = [], [], []
            for i, sen_ids in enumerate(ids):
                out, local_ids = [], sen_ids.clone()
                mask = local_ids != -100
                offset = local_ids[mask].min()
                global_offsets.append(offset)
                local_ids[mask] -= offset
                n_sent = local_ids.max() + 1
                num_sents.append(n_sent)

                for j in range(int(n_sent)):
                    out.append(seq_out[i, local_ids == j].mean(dim=-2, keepdim=True))

                if max_sentences - n_sent:
                    padding = torch.zeros(
                        (int(max_sentences - n_sent), d_model),
                        device=seq_out.device,
                    )
                    out.append(padding)
                agg_out.append(torch.cat(out, dim=0))
            return torch.stack(agg_out), global_offsets, num_sents

        agg_output, offsets, num_sents_item = _get_agg_output(
            sentence_ids, sequence_output
        )
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]

        def _get_preds(pp, offs, num_s, threshold=0.5, alpha=0.05):
            preds = []
            for p, off, ns in zip(pp, offs, num_s):
                rel_probs = p[:ns]
                hits = (rel_probs >= threshold).int()
                if hits.sum() == 0 and rel_probs.max().item() >= alpha:
                    hits[rel_probs.argmax()] = 1
                preds.append(torch.where(hits == 1)[0] + off)
            return preds

        return tuple(_get_preds(probs, offsets, num_sents_item))


def build_sentence_ids(document_sentences):
    """Flatten a list of sentences into token words and per-word sentence ids."""

    words, word_level_sentence_ids = [], []
    for sent_idx, sentence in enumerate(document_sentences):
        tokens = sentence.split()
        words.extend(tokens)
        word_level_sentence_ids.extend([sent_idx] * len(tokens))
    return words, word_level_sentence_ids


def prepare_highlighter_inputs(
    tokenizer,
    query,
    document_sentences,
    max_seq_length=510,
    stride=128,
):
    """Tokenize a single query/document pair and return model-ready tensors."""

    words, word_level_sentence_ids = build_sentence_ids(document_sentences)

    tokenized = tokenizer(
        [query],
        [words],
        truncation="only_second",
        max_length=max_seq_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False,
        is_split_into_words=True,
    )

    word_ids = tokenized.word_ids(0)
    sequence_ids = tokenized.sequence_ids(0)

    token_start_index = 0
    while sequence_ids[token_start_index] != 1:
        token_start_index += 1

    sentence_ids = [-100] * token_start_index
    for word_idx in word_ids[token_start_index:]:
        if word_idx is not None:
            sentence_ids.append(word_level_sentence_ids[word_idx])
        else:
            sentence_ids.append(-100)

    input_ids = tokenized["input_ids"][0][:max_seq_length]
    token_type_ids = tokenized["token_type_ids"][0][:max_seq_length]
    attention_mask = tokenized["attention_mask"][0][:max_seq_length]
    sentence_ids = sentence_ids[:max_seq_length]

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
        "sentence_ids": torch.tensor([sentence_ids], dtype=torch.long),
    }
