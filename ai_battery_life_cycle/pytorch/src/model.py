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
    """Two-layer GRU regressor that maps (batch, seq, features) -> (batch, 1).

    Uses GRUCell instead of nn.GRU to avoid torch_xla patching nn.GRU with
    a scan-based implementation that requires XLA tensors even during CPU runs.
    """

    def __init__(
        self, input_size: int = 18, hidden_size: int = 128, num_layers: int = 2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList(
            [
                nn.GRUCell(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(hidden_size, 1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Remap nn.GRU weight keys to nn.GRUCell keys so pretrained checkpoints load correctly.
        remapped = {}
        for k, v in state_dict.items():
            for param in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                for layer_idx in range(self.num_layers):
                    old_key = f"gru.{param}_l{layer_idx}"
                    new_key = f"gru_cells.{layer_idx}.{param}"
                    if k == old_key:
                        remapped[new_key] = v
                        break
                else:
                    continue
                break
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = [
            torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            for _ in range(self.num_layers)
        ]
        for t in range(seq_len):
            inp = x[:, t, :]
            for layer_idx, cell in enumerate(self.gru_cells):
                h[layer_idx] = cell(inp, h[layer_idx])
                inp = h[layer_idx]
        return self.fc(h[-1])
