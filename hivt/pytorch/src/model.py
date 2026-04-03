# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiVT — Hierarchical Vector Transformer for vehicle trajectory prediction.

Reference: https://github.com/ZikangZhou/HiVT
"HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction" (CVPR 2022).

Architecture:
  1. Local Encoder: per-agent temporal attention over recent history
  2. Global Interactor: cross-agent attention (AAL — agent–agent layer)
  3. Decoder: multi-modal trajectory decoder with K future modes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LocalEncoder(nn.Module):
    """Encodes each agent's motion history independently using self-attention.

    Input per agent: T timesteps of (dx, dy) displacements = 2 features.
    Output: per-agent embedding of size hidden_dim.
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, T, 2) — N agents, T timesteps, (dx, dy)
        Returns:
            (B, N, hidden_dim)
        """
        B, N, T, D = x.shape
        x_flat = x.view(B * N, T, D)
        h = self.pos_enc(self.input_proj(x_flat))
        h = self.transformer(h)          # (B*N, T, hidden_dim)
        h = h[:, -1]                     # take last timestep as agent embedding
        h = self.out_proj(h)
        return h.view(B, N, -1)


class GlobalInteractor(nn.Module):
    """Cross-agent interaction using multi-head attention (AAL layer).

    Models social interactions between all agents in the scene.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

    def forward(self, agent_embeddings):
        """
        Args:
            agent_embeddings: (B, N, hidden_dim)
        Returns:
            (B, N, hidden_dim)
        """
        return self.transformer(agent_embeddings)


class TrajectoryDecoder(nn.Module):
    """Multi-modal trajectory decoder.

    Predicts K possible future trajectories + their probability scores.
    """

    def __init__(self, hidden_dim: int = 64, future_steps: int = 30, num_modes: int = 6):
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps

        # Separate MLPs for each mode (simplified: shared trunk + mode-specific head)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Per-mode trajectory prediction
        self.traj_head = nn.Linear(hidden_dim, num_modes * future_steps * 2)
        # Mode probability logits
        self.prob_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, h):
        """
        Args:
            h: (B, hidden_dim) — embedding of the target agent
        Returns:
            trajectories: (B, num_modes, future_steps, 2)
            probabilities: (B, num_modes)
        """
        B = h.shape[0]
        h = self.trunk(h)
        traj = self.traj_head(h).view(B, self.num_modes, self.future_steps, 2)
        prob = F.softmax(self.prob_head(h), dim=-1)
        return traj, prob


class HiVT(nn.Module):
    """HiVT: Hierarchical Vector Transformer for multi-agent trajectory prediction.

    Reference: https://github.com/ZikangZhou/HiVT
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_heads: int = 4,
        local_layers: int = 2,
        global_layers: int = 2,
        future_steps: int = 30,
        num_modes: int = 6,
    ):
        super().__init__()
        self.local_encoder = LocalEncoder(in_dim, hidden_dim, num_heads, local_layers)
        self.global_interactor = GlobalInteractor(hidden_dim, num_heads, global_layers)
        self.decoder = TrajectoryDecoder(hidden_dim, future_steps, num_modes)

    def forward(self, agent_history, target_idx=None):
        """
        Args:
            agent_history: (B, N, T, 2) — history of N agents over T timesteps (dx, dy)
            target_idx:    int — index of the target agent to predict (default: 0)
        Returns:
            trajectories:  (B, num_modes, future_steps, 2)
            probabilities: (B, num_modes)
        """
        if target_idx is None:
            target_idx = 0

        # Local encoding: each agent independently
        local_feats = self.local_encoder(agent_history)  # (B, N, hidden_dim)

        # Global interaction: agents attend to each other
        global_feats = self.global_interactor(local_feats)  # (B, N, hidden_dim)

        # Decode trajectory for the target agent
        target_feat = global_feats[:, target_idx]  # (B, hidden_dim)
        trajectories, probabilities = self.decoder(target_feat)

        return trajectories, probabilities
