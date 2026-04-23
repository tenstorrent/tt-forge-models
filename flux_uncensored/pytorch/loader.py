#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX Uncensored LoRA model loader implementation.

Loads the FLUX.1-dev base pipeline and applies the uncensored LoRA weights
from enhanceaiteam/Flux-uncensored for text-to-image generation.

Available variants:
- UNCENSORED: Uncensored LoRA applied to FLUX.1-dev
"""

import os
from typing import Any, Optional

import torch
from diffusers import FluxPipeline
from diffusers.models import FluxTransformer2DModel

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "enhanceaiteam/Flux-uncensored"
LORA_WEIGHT_NAME = "lora.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX Uncensored LoRA variants."""

    UNCENSORED = "Uncensored"


class ModelLoader(ForgeModel):
    """FLUX Uncensored LoRA model loader."""

    _VARIANTS = {
        ModelVariant.UNCENSORED: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNCENSORED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe: Optional[FluxPipeline] = None
        self.transformer: Optional[FluxTransformer2DModel] = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_UNCENSORED_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipe = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        self.pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME)

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            dtype = dtype_override if dtype_override is not None else torch.bfloat16
            self.transformer = FluxTransformer2DModel(
                patch_size=1,
                in_channels=64,
                num_layers=19,
                num_single_layers=38,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=True,
            ).to(dtype)
            return self.transformer

        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            if self.transformer is None:
                self.load_model(dtype_override=dtype_override)

            dtype = dtype_override if dtype_override is not None else torch.bfloat16
            config = self.transformer.config
            max_sequence_length = 256
            height = 128
            width = 128
            vae_scale_factor = 8
            num_channels_latents = config.in_channels // 4

            height_latent = 2 * (int(height) // (vae_scale_factor * 2))
            width_latent = 2 * (int(width) // (vae_scale_factor * 2))
            h_packed = height_latent // 2
            w_packed = width_latent // 2

            latents = torch.randn(
                batch_size, h_packed * w_packed, num_channels_latents * 4, dtype=dtype
            )
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
            guidance = (
                torch.full([batch_size], self.guidance_scale, dtype=dtype)
                if config.guidance_embeds
                else None
            )

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

        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        prompt = "An astronaut riding a green horse"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        max_sequence_length = 256
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

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

        guidance = (
            torch.full([batch_size], self.guidance_scale, dtype=dtype)
            if do_classifier_free_guidance
            else None
        )

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
