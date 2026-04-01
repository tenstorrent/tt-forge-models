# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac Sim / IsaacLab Policy Network — RSL-RL Actor-Critic.

Reference: https://github.com/isaac-sim/IsaacSim
           https://github.com/leggedrobotics/rsl_rl

Isaac Sim is a physics simulation platform with heavy CUDA dependencies
(Omniverse, PhysX, Warp, RTX rendering). The learnable component is the
RSL-RL actor-critic policy network used for robot locomotion and manipulation
inference in IsaacLab workflows.

This module ports the inference-only portion of the RSL-RL MLP-based
actor-critic: a standard MLP backbone with ELU activations, matching the
architecture used in IsaacLab PPO training (rsl_rl.models.MLPModel).

CUDA rewrite notes:
  - No CUDA / Omniverse / Warp dependencies are needed at inference time.
  - The simulation (physics, sensors, rendering) runs outside this network.
  - Pure PyTorch — runs on CPU and TT devices without modification.
"""

import torch
import torch.nn as nn


class RslRlMLP(nn.Sequential):
    """MLP backbone matching rsl_rl.modules.mlp.MLP.

    Architecture: Linear(in→h0)+ELU, Linear(h0→h1)+ELU, ..., Linear(h[-1]→out)
    Last layer is linear with no activation, matching RSL-RL's default.

    Reference: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/modules/mlp.py
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-1], output_dim))
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)


class IsaacSimActorCritic(nn.Module):
    """Actor-Critic policy network for Isaac Sim locomotion / manipulation tasks.

    Matches rsl_rl.models.MLPModel used in IsaacLab PPO training pipelines.
    The actor outputs deterministic action means; the critic outputs state values.

    Variants (from IsaacLab task configs):
      - AnymalC_Rough: obs=235 (proprioception + height scan), action=12, hidden=[512,256,128]
      - AnymalC_Flat:  obs=48  (proprioception only),           action=12, hidden=[128,128,128]
      - H1_Velocity:   obs=47  (Unitree H1 humanoid flat),      action=19, hidden=[512,256,128]

    References:
      - IsaacLab ANYmal-B config: isaaclab_tasks/.../anymal_b/agents/rsl_rl_ppo_cfg.py
      - RSL-RL PPO: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
    ):
        super().__init__()
        # Actor: obs → action mean (deterministic at inference)
        self.actor = RslRlMLP(obs_dim, action_dim, actor_hidden_dims)
        # Critic: obs → state value estimate
        self.critic = RslRlMLP(obs_dim, 1, critic_hidden_dims)

    def forward(self, observations: torch.Tensor):
        """
        Args:
            observations: (B, obs_dim) — robot proprioceptive state vector
        Returns:
            action_mean: (B, action_dim) — deterministic action (used at inference)
            value:       (B, 1)          — state value estimate
        """
        action_mean = self.actor(observations)
        value = self.critic(observations)
        return action_mean, value
