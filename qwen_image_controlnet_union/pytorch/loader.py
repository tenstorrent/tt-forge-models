# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image ControlNet Union model loader implementation for conditional image generation.

Loads the Qwen-Image ControlNet Union model from InstantX, which supports
multiple control conditions (Canny, Soft Edge, Depth, Pose) in a single
unified model based on the Qwen-Image diffusion transformer architecture.

Repository: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
"""

import torch
from diffusers import QwenImageControlNetModel
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

CONTROLNET_REPO_ID = "InstantX/Qwen-Image-ControlNet-Union"


class ModelVariant(StrEnum):
    """Available Qwen-Image ControlNet Union model variants."""

    CONTROLNET_UNION = "ControlNet_Union"


class ModelLoader(ForgeModel):
    """Qwen-Image ControlNet Union model loader for conditional image generation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION: ModelConfig(
            pretrained_model_name=CONTROLNET_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-Image ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen-Image ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen-Image ControlNet model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        repo_id = self._variant_config.pretrained_model_name
        self.controlnet = QwenImageControlNetModel.from_pretrained(
            repo_id, torch_dtype=compute_dtype
        )

        return self.controlnet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen-Image ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size (default: 1).

        Returns:
            dict: Input tensors for the ControlNet model.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        in_channels = config.in_channels
        joint_attention_dim = config.joint_attention_dim

        height = 128
        width = 128
        vae_scale_factor = 8
        patch_size = config.patch_size

        h_latent = height // vae_scale_factor
        w_latent = width // vae_scale_factor
        h_patched = h_latent // patch_size
        w_patched = w_latent // patch_size
        seq_len = h_patched * w_patched

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)

        text_seq_len = 128
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        timestep = torch.tensor([500], dtype=torch.long).expand(batch_size)

        controlnet_cond = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": 1.0,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_shapes": [(1, h_patched, w_patched)],
        }

        return inputs
