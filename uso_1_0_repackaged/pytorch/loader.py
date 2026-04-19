#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
USO 1.0 Repackaged model loader implementation.

Loads a FLUX.1-dev transformer from non-gated FP8 weights (Kijai/flux-fp8)
and applies USO (Unified Style-Subject Optimized) LoRA weights from
Comfy-Org/USO_1.0_Repackaged for style/subject-driven text-to-image generation.

Available variants:
- USO_1_0_LORA: FLUX.1-dev (FP8) with USO LoRA applied
"""

import json
import os
import tempfile
from typing import Any, Optional

import torch
from diffusers.models import FluxTransformer2DModel

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

FP8_WEIGHTS_URL = (
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e4m3fn.safetensors"
)
LORA_REPO = "Comfy-Org/USO_1.0_Repackaged"
LORA_FILE = "split_files/loras/uso-flux1-dit-lora-v1.safetensors"

_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available USO 1.0 Repackaged model variants."""

    USO_1_0_LORA = "1.0_LoRA"


class ModelLoader(ForgeModel):
    """USO 1.0 Repackaged model loader using FLUX.1-dev (FP8) with LoRA."""

    _VARIANTS = {
        ModelVariant.USO_1_0_LORA: ModelConfig(
            pretrained_model_name="Kijai/flux-fp8",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.USO_1_0_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="USO_1_0_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def _load_transformer(self, dtype: torch.dtype = torch.float32):
        config_dir = self._make_local_config_dir()
        self._transformer = FluxTransformer2DModel.from_single_file(
            FP8_WEIGHTS_URL,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.load_lora_adapter(
            LORA_REPO,
            weight_name=LORA_FILE,
        )
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)
        return self._transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._transformer is None:
            self._load_transformer(dtype)

        max_sequence_length = 256
        num_images_per_prompt = 1
        height = 128
        width = 128
        num_channels_latents = self._transformer.config.in_channels // 4
        vae_scale_factor = 8

        pooled_projection_dim = self._transformer.config.pooled_projection_dim
        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt, pooled_projection_dim, dtype=dtype
        )

        joint_attention_dim = self._transformer.config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            joint_attention_dim,
            dtype=dtype,
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        latents = torch.randn(shape, dtype=dtype)
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        guidance = torch.full([batch_size], 3.5, dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
