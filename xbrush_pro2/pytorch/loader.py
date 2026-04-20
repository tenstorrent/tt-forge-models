# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XBrush-Pro2 model loader implementation for text-to-image generation.

Uses a local transformer config to avoid accessing the gated
lightweight/XBrush-Pro2 HuggingFace repo.
"""

import torch
from diffusers.models import QwenImageTransformer2DModel
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

_TRANSFORMER_CONFIG = {
    "_class_name": "QwenImageTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": [16, 56, 56],
    "zero_cond_t": False,
    "use_additional_t_cond": False,
    "use_layer3d_rope": False,
}


class ModelVariant(StrEnum):
    """Available XBrush-Pro2 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """XBrush-Pro2 model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="lightweight/XBrush-Pro2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="XBrush-Pro2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.transformer = QwenImageTransformer2DModel(**_TRANSFORMER_CONFIG).to(dtype)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        patch_size = config.patch_size
        in_channels = config.in_channels

        # Compute latent spatial dimensions (after VAE + patchification)
        vae_scale_factor = 8
        h_latent = height // vae_scale_factor
        w_latent = width // vae_scale_factor
        h_patches = h_latent // patch_size
        w_patches = w_latent // patch_size
        image_seq_len = h_patches * w_patches

        # Hidden states: (B, image_seq_len, in_channels)
        hidden_states = torch.randn(batch_size, image_seq_len, in_channels, dtype=dtype)

        # Encoder hidden states (text embeddings): (B, text_seq_len, joint_attention_dim)
        text_seq_len = 128
        joint_attention_dim = config.joint_attention_dim
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        # Image shapes for RoPE computation: list of tuples (t, h, w)
        img_shapes = [[(1, h_patches, w_patches)] for _ in range(batch_size)]

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }

        return inputs
