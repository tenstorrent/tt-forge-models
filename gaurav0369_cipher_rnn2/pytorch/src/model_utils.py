# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom RNN architecture matching the Gaurav0369/cipher-rnn2 checkpoint layout.

The published checkpoint stores a hand-rolled stacked RNN (not ``nn.RNN``)
whose parameters live in ``ParameterList``s keyed by layer index
(``Wxh.0``/``Wxh.1``, ``Whh.0``/``Whh.1``, ``bh.0``/``bh.1``) plus an
``embedding.weight`` and an output projection (``Why``/``by``).
"""

import torch
import torch.nn as nn


class CipherRNN(nn.Module):
    """Stacked tanh RNN that maps plaintext token ids to cipher-token logits."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        Wxh = []
        input_dim = embedding_dim
        for _ in range(num_layers):
            Wxh.append(nn.Parameter(torch.empty(hidden_size, input_dim)))
            input_dim = hidden_size
        self.Wxh = nn.ParameterList(Wxh)

        self.Whh = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bh = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size)) for _ in range(num_layers)]
        )

        self.Why = nn.Parameter(torch.empty(output_size, hidden_size))
        self.by = nn.Parameter(torch.empty(output_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        embedded = self.embedding(input_ids)
        dtype = embedded.dtype
        device = embedded.device

        h = [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for layer in range(self.num_layers):
                h_new = torch.tanh(
                    x_t @ self.Wxh[layer].t()
                    + h[layer] @ self.Whh[layer].t()
                    + self.bh[layer]
                )
                h[layer] = h_new
                x_t = h_new
            outputs.append(x_t)

        hidden_seq = torch.stack(outputs, dim=1)
        return hidden_seq @ self.Why.t() + self.by
