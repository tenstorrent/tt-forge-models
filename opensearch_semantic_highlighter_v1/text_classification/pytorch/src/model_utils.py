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
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sentence_ids=None,
        max_sentences=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.dropout(outputs[0])
        agg_output = _aggregate_by_sentence(sentence_ids, sequence_output, max_sentences)
        logits = self.classifier(agg_output)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]
        return probs


def _aggregate_by_sentence(ids, seq_out, max_sents):
    """
    Vectorized sentence aggregation without D2H transfers.

    ids: [B, S] — sentence IDs (-100 for padding, 0-based otherwise)
    seq_out: [B, S, D]
    max_sents: Python int (precomputed on CPU, avoids device→host transfer)

    Returns: [B, max_sents, D] mean embedding per sentence (zero for empty slots)
    """
    B, S, D = seq_out.shape
    valid_mask = ids != -100  # [B, S]

    # Per-batch minimum valid sentence ID (offset to normalise to 0-based)
    ids_for_min = torch.where(valid_mask, ids, torch.full_like(ids, max_sents))
    offsets = ids_for_min.min(dim=1).values  # [B]

    local_ids = ids - offsets.unsqueeze(1)  # [B, S]
    local_ids = torch.where(valid_mask, local_ids, torch.full_like(local_ids, -1))

    # One-hot membership: sent_mask[b, s, j] = 1 iff token s → sentence j
    j = torch.arange(max_sents, device=ids.device)  # [max_sents]
    sent_mask = (local_ids.unsqueeze(-1) == j).float()  # [B, S, max_sents]

    sent_sum = torch.einsum("bsd,bsj->bjd", seq_out, sent_mask)  # [B, max_sents, D]
    sent_count = sent_mask.sum(dim=1)  # [B, max_sents]
    return sent_sum / sent_count.unsqueeze(-1).clamp(min=1.0)


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
        [query.split()],
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
        "max_sentences": len(document_sentences),
    }
