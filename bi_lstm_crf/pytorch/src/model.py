# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiLSTM-CRF model.
"""

import torch
import torch.nn as nn

IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """CRF with tensorized Viterbi decode (no .item() or Python loops)."""

    def __init__(self, in_features: int, num_tags: int):
        super().__init__()
        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)
        self.transitions = nn.Parameter(
            torch.randn(self.num_tags, self.num_tags), requires_grad=True
        )
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(
        self, features: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode tags. Returns (best_score [B], best_paths [B, L])."""
        features = self.fc(features)
        masks = masks[:, : features.size(1)].float()
        return self._viterbi_decode(features, masks)

    def _viterbi_decode(
        self, features: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fully tensorized Viterbi decode. No .item() or numpy."""
        B, L, C = features.shape
        device = features.device
        dtype = features.dtype

        bps = torch.zeros(B, L, C, dtype=torch.long, device=device)
        max_score = torch.full((B, C), IMPOSSIBLE, device=device, dtype=dtype)
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)
            emit_score_t = features[:, t]
            acc_score_t = max_score.unsqueeze(1) + self.transitions
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t = acc_score_t + emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)

        max_score = max_score + self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Tensorized back-trace: follow back pointers without .item()
        best_tags = best_tag.clone()
        best_paths = torch.zeros(B, L, dtype=torch.long, device=device)
        best_paths[:, L - 1] = best_tag

        for t in range(L - 2, -1, -1):
            prev_tags = torch.gather(
                bps[:, t + 1, :], 1, best_tags.unsqueeze(1)
            ).squeeze(1)
            best_tags = torch.where(masks[:, t].bool(), prev_tags, best_tags)
            best_paths[:, t] = best_tags

        return best_score, best_paths


class BiRnnCrf(nn.Module):
    """BiLSTM-CRF without pack_padded_sequence (torch.compile/XLA compatible)."""

    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_rnn_layers: int = 1,
        rnn: str = "lstm",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.crf = CRF(hidden_dim, tagset_size)

    def _build_features(
        self, sentences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build LSTM features. Uses padded LSTM (no pack_padded_sequence) for XLA compatibility."""
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())
        lstm_out, _ = self.rnn(embeds)
        return lstm_out, masks

    def forward(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (best_score [B], best_paths [B, L])."""
        features, masks = self._build_features(xs)
        return self.crf(features, masks)
