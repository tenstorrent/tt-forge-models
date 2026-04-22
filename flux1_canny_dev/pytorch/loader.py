# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Canny-dev model loader implementation for Canny-edge-conditioned image generation.

Loads a FluxTransformer2DModel with in_channels=128 (latents + control concatenated)
using the FLUX.1-dev architecture and random weights, avoiding the gated HF repo.
"""

import torch
from diffusers import FluxTransformer2DModel
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

# FLUX.1-dev transformer config; Canny-dev doubles in_channels to 128
# so the concatenated [latents, control_latents] can be passed as hidden_states.
_FLUX_CANNY_CONFIG = {
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
    "axes_dims_rope": [16, 56, 56],
}


class ModelVariant(StrEnum):
    """Available FLUX.1-Canny-dev model variants."""

    CANNY_DEV = "Canny_Dev"


class ModelLoader(ForgeModel):
    """FLUX.1-Canny-dev model loader for Canny-edge-conditioned image generation tasks."""

    _VARIANTS = {
        ModelVariant.CANNY_DEV: ModelConfig(
            pretrained_model_name="black-forest-labs/FLUX.1-Canny-dev",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CANNY_DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 30.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.1-Canny-dev",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.transformer = FluxTransformer2DModel(**_FLUX_CANNY_CONFIG)
        self.transformer = self.transformer.to(dtype=dtype)
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        vae_scale_factor = 8
        max_sequence_length = 256

        # num_channels_latents for control pipeline: in_channels // 8
        num_channels_latents = config.in_channels // 8

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Each of latents and control_image has shape (B, h*w, C)
        latents = torch.randn(
            batch_size,
            h_packed * w_packed,
            num_channels_latents * 4,
            dtype=dtype,
        )
        control_image = torch.randn_like(latents)

        # Concatenate along channel dim as FluxControlPipeline does
        hidden_states = torch.cat([latents, control_image], dim=2)

        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        if self.guidance_scale > 1.0:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
