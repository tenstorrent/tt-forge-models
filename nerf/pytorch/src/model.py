# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vanilla NeRF (Neural Radiance Fields) model implementation.

Reference: https://github.com/Yubel426/NeRF-3DGS-at-CVPR-2024
Original NeRF paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
(Mildenhall et al., ECCV 2020)

Architecture:
  - Positional encoding for 3D location and view direction
  - 8-layer MLP for density (sigma) and feature prediction
  - Additional 1-layer branch for color (RGB) conditioned on view direction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding for NeRF inputs.

    Encodes a D-dimensional input as [sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)].
    """

    def __init__(self, input_dim: int, num_freqs: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)
        self.output_dim = input_dim + 2 * input_dim * num_freqs

    def forward(self, x):
        """
        Args:
            x: (..., input_dim)
        Returns:
            encoded: (..., output_dim)
        """
        out = [x]
        for freq in self.freqs:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class NeRFMLP(nn.Module):
    """Vanilla NeRF MLP.

    Takes 3D position and view direction as input, outputs RGB color and density.

    Architecture follows the original NeRF paper:
      - 8 fully-connected layers (width=256) with ReLU
      - Skip connection at layer 4 (concatenate input again)
      - Density (sigma) predicted from position-only features
      - Color (RGB) predicted from position features + view direction
    """

    def __init__(
        self,
        pos_freq: int = 10,
        dir_freq: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(3, pos_freq)
        self.dir_enc = PositionalEncoding(3, dir_freq)

        pos_dim = self.pos_enc.output_dim
        dir_dim = self.dir_enc.output_dim

        # Main MLP (layers 1-8 with skip at layer 4)
        self.fc1 = nn.Linear(pos_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim + pos_dim, hidden_dim)  # skip connection
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)

        # Density head (no activation — raw sigma, apply relu externally)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Feature for color
        self.feat_head = nn.Linear(hidden_dim, hidden_dim)

        # Color MLP (conditioned on view direction)
        self.color_fc = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)

    def forward(self, positions, directions):
        """
        Args:
            positions:  (..., 3) — 3D sample positions along rays
            directions: (..., 3) — normalized viewing directions (broadcast or same shape)
        Returns:
            rgb:   (..., 3) in [0, 1]
            sigma: (..., 1) density >= 0
        """
        pos_enc = self.pos_enc(positions)
        dir_enc = self.dir_enc(directions)

        h = F.relu(self.fc1(pos_enc))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(torch.cat([h, pos_enc], dim=-1)))  # skip
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))

        sigma = F.relu(self.sigma_head(h))

        feat = self.feat_head(h)
        color_input = torch.cat([feat, dir_enc], dim=-1)
        rgb = torch.sigmoid(self.rgb_head(F.relu(self.color_fc(color_input))))

        return rgb, sigma
