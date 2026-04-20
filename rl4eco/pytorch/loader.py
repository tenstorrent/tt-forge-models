# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RL4Eco model loader implementation.

Loads a Proximal Policy Optimization (PPO) agent trained on the
rl4fisheries Age-Structured Model (AsmEnv) environment with the
two-observation variant. The agent is distributed as a
stable-baselines3 checkpoint in the boettiger-lab/rl4eco HuggingFace
repository and uses an MLP policy to map observations to a
continuous harvest action.
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
    """Available RL4Eco model variants."""

    PPO_ASM2O_V0 = "ppo_asm2o_v0"


class ModelLoader(ForgeModel):
    """RL4Eco model loader implementation."""

    _VARIANTS = {
        ModelVariant.PPO_ASM2O_V0: ModelConfig(
            pretrained_model_name="boettiger-lab/rl4eco",
        ),
    }

    # Subpath within the HuggingFace repo for each variant's SB3 checkpoint.
    _CHECKPOINTS = {
        ModelVariant.PPO_ASM2O_V0: "sb3/PPO-Asm2o-v0-1.zip",
    }

    # Observation dimensionality for each variant (AsmEnv with observe_2o).
    _OBS_DIMS = {
        ModelVariant.PPO_ASM2O_V0: 2,
    }

    DEFAULT_VARIANT = ModelVariant.PPO_ASM2O_V0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RL4Eco",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RL4Eco PPO policy network.

        Uses stable-baselines3 to load the pretrained PPO checkpoint from
        HuggingFace, then extracts the policy network as a standard
        PyTorch module.

        Args:
            dtype_override: Optional torch.dtype to override the model's
                default dtype.

        Returns:
            torch.nn.Module: The PPO policy network in eval mode.
        """
        from huggingface_hub import hf_hub_download
        from stable_baselines3 import PPO

        pretrained_model_name = self._variant_config.pretrained_model_name
        checkpoint_filename = self._CHECKPOINTS[self._variant]

        model_path = hf_hub_download(
            repo_id=pretrained_model_name,
            filename=checkpoint_filename,
        )
        # SB3 re-appends the .zip extension during load.
        model_path = model_path.removesuffix(".zip")

        # Override schedule-style entries that may fail to unpickle
        # when SB3 versions differ between training and inference.
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        ppo_model = PPO.load(model_path, custom_objects=custom_objects)

        policy = ppo_model.policy
        policy.eval()

        if dtype_override is not None:
            policy = policy.to(dtype_override)

        return policy

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RL4Eco PPO policy.

        AsmEnv observations are bounded floats in [-1, 1]; the observe_2o
        variant produces a 2-dimensional observation vector.

        Args:
            dtype_override: Optional torch.dtype to override the inputs'
                default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Observation tensor of shape (batch_size, obs_dim).
        """
        obs_dim = self._OBS_DIMS[self._variant]
        obs = torch.zeros((batch_size, obs_dim), dtype=torch.float32)

        if dtype_override is not None:
            obs = obs.to(dtype_override)

        return obs
