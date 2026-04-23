#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies the Castor-3D-Sketchfab LoRA
from prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA for 3D Sketchfab style
text-to-image generation.

Repository: https://huggingface.co/prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA
"""

import os
from typing import Any, Optional

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA"
LORA_WEIGHT_NAME = "Castor-3D-Sketchfab-Flux-LoRA.safetensors"

# Known FLUX.1-dev transformer configuration for random-weights mode
_FLUX_DEV_CONFIG = {
    "in_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
}

# Synthetic input dimensions matching a 128x128 image through FLUX pipeline
_HEIGHT = 128
_WIDTH = 128
_VAE_SCALE_FACTOR = 8
_MAX_SEQ_LEN = 256


class ModelVariant(StrEnum):
    """Available prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA variants."""

    SKETCHFAB_3D = "Sketchfab3D"


class ModelLoader(ForgeModel):
    """prithivMLmods/Castor-3D-Sketchfab-Flux-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.SKETCHFAB_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SKETCHFAB_3D

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Castor-3D-Sketchfab-Flux-LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _use_random_weights() -> bool:
        return bool(
            os.environ.get("TT_RANDOM_WEIGHTS")
            or os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC")
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """Load the FLUX.1-dev transformer with Castor-3D-Sketchfab LoRA weights.

        Returns:
            FluxTransformer2DModel with LoRA weights applied (or random weights in
            compile-only mode).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self._use_random_weights():
            transformer = FluxTransformer2DModel(**_FLUX_DEV_CONFIG)
            transformer = transformer.to(dtype)
            return transformer

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare transformer inputs for 3D Sketchfab style text-to-image generation.

        Returns:
            dict with tensor inputs for FluxTransformer2DModel.forward().
        """
        if self._use_random_weights() or self.pipeline is None:
            return self._synthetic_inputs()
        return self._encoded_inputs()

    def _synthetic_inputs(self) -> dict:
        dtype = torch.float32
        in_channels = _FLUX_DEV_CONFIG["in_channels"]
        num_channels_latents = in_channels // 4

        height_latent = 2 * (_HEIGHT // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (_WIDTH // (_VAE_SCALE_FACTOR * 2))

        h_packed = height_latent // 2
        w_packed = width_latent // 2
        seq_img = h_packed * w_packed
        seq_txt = _MAX_SEQ_LEN
        joint_dim = _FLUX_DEV_CONFIG["joint_attention_dim"]
        pooled_dim = _FLUX_DEV_CONFIG["pooled_projection_dim"]

        latents = torch.randn(1, seq_img, num_channels_latents * 4, dtype=dtype)

        img_ids = torch.zeros(seq_img, 3, dtype=dtype)
        img_ids[:, 1] = torch.arange(seq_img, dtype=dtype) // w_packed
        img_ids[:, 2] = torch.arange(seq_img, dtype=dtype) % w_packed

        txt_ids = torch.zeros(seq_txt, 3, dtype=dtype)

        return {
            "hidden_states": latents,
            "encoder_hidden_states": torch.randn(1, seq_txt, joint_dim, dtype=dtype),
            "pooled_projections": torch.randn(1, pooled_dim, dtype=dtype),
            "timestep": torch.tensor([1.0], dtype=dtype),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": torch.tensor([3.5], dtype=dtype),
        }

    def _encoded_inputs(self) -> dict:
        prompt = (
            "3D Sketchfab, a stylized low-poly fox standing on a grassy "
            "hill, bright studio lighting, turntable render"
        )
        dtype = torch.float32
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        height_latent = 2 * (_HEIGHT // (self.pipeline.vae_scale_factor * 2))
        width_latent = 2 * (_WIDTH // (self.pipeline.vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2
        seq_img = h_packed * w_packed

        clip_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_projections = self.pipeline.text_encoder(
            clip_inputs.input_ids, output_hidden_states=False
        ).pooler_output.to(dtype=dtype)

        t5_inputs = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=_MAX_SEQ_LEN,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipeline.text_encoder_2(
            t5_inputs.input_ids, output_hidden_states=False
        )[0].to(dtype=dtype)

        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3, dtype=dtype)

        latents = torch.randn(
            1, num_channels_latents, height_latent, width_latent, dtype=dtype
        )
        latents = latents.view(1, num_channels_latents, h_packed, 2, w_packed, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(1, seq_img, num_channels_latents * 4)

        img_ids = torch.zeros(h_packed, w_packed, 3, dtype=dtype)
        img_ids[..., 1] = torch.arange(h_packed, dtype=dtype)[:, None]
        img_ids[..., 2] = torch.arange(w_packed, dtype=dtype)[None, :]
        img_ids = img_ids.reshape(-1, 3)

        return {
            "hidden_states": latents,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": torch.tensor([3.5], dtype=dtype),
        }
