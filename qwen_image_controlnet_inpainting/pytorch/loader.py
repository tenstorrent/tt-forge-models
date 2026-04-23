# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image ControlNet Inpainting model loader implementation for conditional image generation.

Loads the Qwen-Image ControlNet Inpainting model from InstantX, which provides
mask-based inpainting and outpainting conditioning for the Qwen-Image diffusion
transformer architecture.

Repository: https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting
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

CONTROLNET_REPO_ID = "InstantX/Qwen-Image-ControlNet-Inpainting"


class ModelVariant(StrEnum):
    """Available Qwen-Image ControlNet Inpainting model variants."""

    CONTROLNET_INPAINTING = "ControlNet_Inpainting"


class ModelLoader(ForgeModel):
    """Qwen-Image ControlNet Inpainting model loader for conditional image generation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_INPAINTING: ModelConfig(
            pretrained_model_name=CONTROLNET_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_INPAINTING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen-Image ControlNet Inpainting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen-Image ControlNet Inpainting model.

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
        """Load and return sample inputs for the Qwen-Image ControlNet Inpainting model.

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

        # Model config: num_attention_heads=24, attention_head_dim=128,
        # joint_attention_dim=3584, in_channels=64, patch_size=2,
        # extra_condition_channels=4, axes_dims_rope=[16, 56, 56]
        hidden_size = config.num_attention_heads * config.attention_head_dim  # 3072
        joint_attention_dim = config.joint_attention_dim  # 3584

        # Image dimensions (small for testing)
        height = 128
        width = 128
        vae_scale_factor = 8
        patch_size = config.patch_size  # 2

        # Latent dimensions after VAE encoding and patchification
        h_latent = height // vae_scale_factor  # 16
        w_latent = width // vae_scale_factor  # 16
        h_patched = h_latent // patch_size  # 8
        w_patched = w_latent // patch_size  # 8
        seq_len = h_patched * w_patched  # 64

        # Hidden states (packed latent representation)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

        # Encoder hidden states (text embeddings)
        text_seq_len = 128
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        # ControlNet conditioning image concatenated with mask channels
        # (in_channels + extra_condition_channels) for inpainting
        controlnet_cond_channels = config.in_channels + config.extra_condition_channels
        controlnet_cond = torch.randn(
            batch_size, controlnet_cond_channels, h_latent, w_latent, dtype=dtype
        )

        # Image shapes for 3D RoPE: list of (frame, height, width) per sample
        img_shapes = [(1, h_patched, w_patched)] * batch_size

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "img_shapes": img_shapes,
            "conditioning_scale": 1.0,
        }

        return inputs
