# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
hoanganho0o/PMTACADEMY model loader implementation.

Loads a FLUX.1-dev architecture transformer and applies the PMT ACADEMY Ultra
Realistic LoRA from hoanganho0o/PMTACADEMY for photorealistic text-to-image
generation. The transformer is instantiated from config to avoid the gated
black-forest-labs/FLUX.1-dev repository.

Repository: https://huggingface.co/hoanganho0o/PMTACADEMY
"""

import os
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

LORA_REPO = "hoanganho0o/PMTACADEMY"
LORA_FILENAME = "PMT_ACADEMY_Ultra Realistic V12.0.safetensors"

FLUX_DEV_TRANSFORMER_CONFIG = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}


class ModelVariant(StrEnum):
    """Available hoanganho0o/PMTACADEMY model variants."""

    ULTRA_REALISTIC = "UltraRealistic"


class ModelLoader(ForgeModel):
    """hoanganho0o/PMTACADEMY model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.ULTRA_REALISTIC: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ULTRA_REALISTIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PMTACADEMY",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype=None):
        """Instantiate FLUX.1-dev transformer from config and optionally apply LoRA."""
        self._transformer = FluxTransformer2DModel(**FLUX_DEV_TRANSFORMER_CONFIG)

        if dtype is not None:
            self._transformer = self._transformer.to(dtype=dtype)

        if not os.environ.get("TT_RANDOM_WEIGHTS"):
            self._transformer.load_lora_adapter(
                LORA_REPO,
                weight_name=LORA_FILENAME,
            )

        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._transformer is None:
            self._load_transformer(dtype=dtype_override)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._transformer is None:
            self._load_transformer(dtype=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        in_channels = config.in_channels
        num_channels_latents = in_channels // 4
        joint_attention_dim = config.joint_attention_dim
        pooled_projection_dim = config.pooled_projection_dim

        height = 128
        width = 128
        vae_scale_factor = 8
        max_sequence_length = 256
        guidance_scale = 3.5

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)
        guidance = torch.full([batch_size], guidance_scale, dtype=dtype)
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
