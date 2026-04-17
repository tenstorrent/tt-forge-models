# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PPO Pendulum-v1 model loader implementation.

Loads a Proximal Policy Optimization (PPO) agent trained on the Pendulum-v1
Gymnasium environment using stable-baselines3. The model uses an MLP policy
to map observations (cos(theta), sin(theta), angular velocity) to a continuous
torque action.
"""
import torch
import torch.nn as nn
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class PPOActorNetwork(nn.Module):
    """Extracts the actor network from an SB3 PPO policy as a clean nn.Module.

    SB3's policy forward pass internally casts inputs to float32, which breaks
    bfloat16 inference. This module bypasses that by directly composing the
    underlying nn.Module components: flatten -> policy_net -> action_net.
    """

    def __init__(self, policy):
        super().__init__()
        self.flatten = policy.features_extractor.flatten
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs):
        x = self.flatten(obs)
        x = self.policy_net(x)
        return self.action_net(x)


class ModelVariant(StrEnum):
    """Available PPO Pendulum-v1 model variants."""

    PPO_PENDULUM_V1 = "Ppo_Pendulum_V1"


class ModelLoader(ForgeModel):
    """PPO Pendulum-v1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.PPO_PENDULUM_V1: ModelConfig(
            pretrained_model_name="HumanCompatibleAI/ppo-Pendulum-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PPO_PENDULUM_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PPO_Pendulum_V1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PPO Pendulum-v1 policy network.

        Uses stable-baselines3 to load the pretrained PPO model from HuggingFace,
        then extracts the policy network as a standard PyTorch module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The PPO policy network.
        """
        from huggingface_sb3 import load_from_hub
        from stable_baselines3 import PPO

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Download and load the SB3 model from HuggingFace
        checkpoint = load_from_hub(
            repo_id=pretrained_model_name,
            filename="ppo-Pendulum-v1.zip",
        )
        sb3_model = PPO.load(checkpoint)

        # Extract the policy network as a PyTorch module
        policy = sb3_model.policy
        policy.eval()

        model = PPOActorNetwork(policy)
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PPO Pendulum-v1 model.

        Pendulum-v1 observations are 3-dimensional:
        [cos(theta), sin(theta), angular_velocity].

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Observation tensor of shape (batch_size, 3).
        """
        # Pendulum-v1 observation space: [cos(theta), sin(theta), angular_velocity]
        # cos and sin are in [-1, 1], angular velocity is in [-8, 8]
        obs = torch.tensor(
            [[0.5, 0.866, -1.0]] * batch_size,
            dtype=torch.float32,
        )

        if dtype_override is not None:
            obs = obs.to(dtype_override)

        return obs
