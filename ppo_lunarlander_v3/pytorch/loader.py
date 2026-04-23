# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PPO LunarLander-v3 model loader implementation.

Loads a Proximal Policy Optimization (PPO) agent trained on the LunarLander-v3
Gymnasium environment using stable-baselines3. The model uses an MLP policy
to map 8-dimensional observations (x, y position; x, y velocity; angle;
angular velocity; left and right leg ground contact) to a discrete action
(no-op, fire left, fire main, fire right engine).
"""
import torch
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


class ModelVariant(StrEnum):
    """Available PPO LunarLander-v3 model variants."""

    PPO_LUNARLANDER_V3 = "Ppo_Lunarlander_V3"


class ModelLoader(ForgeModel):
    """PPO LunarLander-v3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.PPO_LUNARLANDER_V3: ModelConfig(
            pretrained_model_name="AllIllusion/LunarLander-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PPO_LUNARLANDER_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PPO_LunarLander_V3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the PPO LunarLander-v3 policy network.

        Uses stable-baselines3 to load the pretrained PPO model from HuggingFace,
        then extracts the policy network as a standard PyTorch module.

        Returns:
            torch.nn.Module: The PPO policy network in float32.
        """
        from huggingface_sb3 import load_from_hub
        from stable_baselines3 import PPO

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Download and load the SB3 model from HuggingFace
        checkpoint = load_from_hub(
            repo_id=pretrained_model_name,
            filename="LunarLander-v3_PPO_TotalStep100K.zip",
        )
        sb3_model = PPO.load(checkpoint)

        # Extract the policy network as a PyTorch module
        policy = sb3_model.policy
        policy.eval()

        return policy

    def load_inputs(self, batch_size=1, **kwargs):
        """Load and return sample inputs for the PPO LunarLander-v3 model.

        LunarLander-v3 observations are 8-dimensional:
        [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact].

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Observation tensor of shape (batch_size, 8).
        """
        # LunarLander-v3 observation space: 8-dimensional state vector.
        # Positions, velocities, angle, angular velocity, and two boolean leg-contact flags.
        return torch.tensor(
            [[0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * batch_size,
            dtype=torch.float32,
        )
