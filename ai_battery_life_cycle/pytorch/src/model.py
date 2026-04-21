# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Architectures for NeerajCodz/aiBatteryLifeCycle deep learning checkpoints.

The HuggingFace repository distributes PyTorch state dicts without the
accompanying module definitions, so the architectures are reconstructed here
to match the parameter shapes exactly.
"""
import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    """Two-layer LSTM regressor that maps (batch, seq, features) -> (batch, 1)."""

    def __init__(
        self, input_size: int = 18, hidden_size: int = 128, num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BidirectionalLSTM(nn.Module):
    """Two-layer bidirectional LSTM regressor with a linear head over the last step."""

    def __init__(
        self, input_size: int = 18, hidden_size: int = 128, num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """Two-layer GRU regressor that maps (batch, seq, features) -> (batch, 1)."""

    def __init__(
        self, input_size: int = 18, hidden_size: int = 128, num_layers: int = 2
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
