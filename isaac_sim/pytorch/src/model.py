# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac Lab actor MLP policy — RSL-RL architecture for locomotion tasks.

Reference: https://github.com/leggedrobotics/rsl_rl
           https://github.com/isaac-sim/IsaacLab

The actor network is a simple MLP that maps observations to actions.
Architecture: obs → [Linear → ELU] × N → Linear → actions

Supported tasks:
  - Isaac-Velocity-Flat-Anymal-C-v0   (obs=48,  act=12, hidden=[128,128,128])
  - Isaac-Velocity-Rough-Anymal-C-v0  (obs=235, act=12, hidden=[512,256,128])
  - Isaac-Velocity-Rough-H1-v0        (obs=256, act=19, hidden=[512,256,128])
"""

import torch
import torch.nn as nn


class IsaacLabActorMLP(nn.Module):
    """Actor MLP from RSL-RL's ActorCritic, used for locomotion policy inference.

    Only the actor network is needed for deployment — the critic and noise std
    are training artifacts that are discarded at inference time.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)
