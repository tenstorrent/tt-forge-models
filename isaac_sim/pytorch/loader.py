# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac Sim / IsaacLab RSL-RL Actor-Critic model loader.

Reference: https://github.com/isaac-sim/IsaacSim
           https://github.com/leggedrobotics/rsl_rl

The neural network (RSL-RL actor-critic MLP) is isolated from the Isaac Sim
simulation infrastructure (Omniverse, PhysX, Warp, CUDA) which is not needed
at inference. Only the policy network runs on the robot.

Variants match published IsaacLab task configurations:
  - AnymalC_Rough: rough-terrain locomotion, obs=235 (proprioception + height scan)
  - AnymalC_Flat:  flat-terrain locomotion,  obs=48  (proprioception only)
  - H1_Velocity:   Unitree H1 humanoid flat, obs=47, action=19
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model import IsaacSimActorCritic


@dataclass
class IsaacSimConfig(ModelConfig):
    """Configuration for an Isaac Sim / IsaacLab RSL-RL policy variant."""

    obs_dim: int = 48
    action_dim: int = 12
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])


class ModelVariant(StrEnum):
    """Available Isaac Sim policy model variants."""

    # ANYmal-C quadruped, rough-terrain velocity tracking
    # obs=235: joint pos(12)+vel(12)+base_lin_vel(3)+ang_vel(3)+gravity(3)+cmd(3)+prev_action(12)+height_scan(187)
    # hidden=[512,256,128] — from IsaacLab rsl_rl_ppo_cfg.py (AnymalBRoughPPORunnerCfg)
    ANYMAL_C_ROUGH = "AnymalC_Rough"

    # ANYmal-C quadruped, flat-terrain velocity tracking
    # obs=48: joint pos(12)+vel(12)+base_lin_vel(3)+ang_vel(3)+gravity(3)+cmd(3)+prev_action(12)
    # hidden=[128,128,128] — from IsaacLab rsl_rl_ppo_cfg.py (AnymalBFlatPPORunnerCfg)
    ANYMAL_C_FLAT = "AnymalC_Flat"

    # Unitree H1 humanoid, flat-terrain velocity tracking
    # obs=47: joint pos(19)+vel(19)+base_ang_vel(3)+projected_gravity(3)+cmd(3)
    # hidden=[512,256,128] — matches IsaacLab H1 velocity task config
    H1_VELOCITY = "H1_Velocity"


class ModelLoader(ForgeModel):
    """Isaac Sim RSL-RL actor-critic policy loader for robotic RL inference."""

    _VARIANTS = {
        ModelVariant.ANYMAL_C_ROUGH: IsaacSimConfig(
            pretrained_model_name="isaac_sim_anymal_c_rough",
            obs_dim=235,
            action_dim=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
        ),
        ModelVariant.ANYMAL_C_FLAT: IsaacSimConfig(
            pretrained_model_name="isaac_sim_anymal_c_flat",
            obs_dim=48,
            action_dim=12,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
        ),
        ModelVariant.H1_VELOCITY: IsaacSimConfig(
            pretrained_model_name="isaac_sim_h1_velocity",
            obs_dim=47,
            action_dim=19,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANYMAL_C_FLAT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="IsaacSim",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Isaac Sim RSL-RL actor-critic policy network.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: IsaacSimActorCritic instance in eval mode.
        """
        cfg = self._variant_config
        model = IsaacSimActorCritic(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample observation vectors for Isaac Sim policy inference.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Number of parallel environments (default: 1).

        Returns:
            torch.Tensor: observations (B, obs_dim) — robot proprioceptive state
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        observations = torch.randn(batch_size, cfg.obs_dim, dtype=dtype)
        return observations
