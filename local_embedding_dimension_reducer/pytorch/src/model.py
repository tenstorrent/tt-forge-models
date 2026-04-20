# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Embedding dimension reducer architecture.

Reconstructed from DialOnce/local-EmbeddingDimensionReducer safetensors weights:
a two-layer MLP that maps ``input_dim`` embeddings through a ``hidden_size``
hidden layer to ``output_dim`` outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class EmbeddingDimensionReducer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        return self.linear(x)
