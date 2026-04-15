#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX Realism LoRA model loader implementation.

Uses from_config with an ungated mirror to avoid gated repo access
for the FLUX.1-dev base model. Returns the transformer component
with synthetic inputs for compile-only testing.

Available variants:
- REALISM: FLUX.1-dev transformer (Realism LoRA config)
"""

from typing import Optional

import torch
from diffusers import FluxTransformer2DModel

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

# Use ungated mirror to load config without gated repo access
UNGATED_MODEL = "camenduru/FLUX.1-dev-ungated"


class ModelVariant(StrEnum):
    """Available FLUX Realism LoRA variants."""

    REALISM = "Realism"


class ModelLoader(ForgeModel):
    """FLUX Realism LoRA model loader."""

    _VARIANTS = {
        ModelVariant.REALISM: ModelConfig(
            pretrained_model_name=UNGATED_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.REALISM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_REALISM_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        config = FluxTransformer2DModel.load_config(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
        )

        self.transformer = FluxTransformer2DModel.from_config(config, **load_kwargs)

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )

        latent_ids = torch.zeros(h_packed, w_packed, 3)
        latent_ids[..., 1] = latent_ids[..., 1] + torch.arange(h_packed)[:, None]
        latent_ids[..., 2] = latent_ids[..., 2] + torch.arange(w_packed)[None, :]
        latent_ids = latent_ids.reshape(-1, 3).to(dtype=dtype)

        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)

        pooled_projection_dim = config.pooled_projection_dim
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)

        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }
