# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX Impressionism LoRA model loader implementation.

Loads a FLUX.1-dev transformer (random weights, no gated HF access needed)
and applies the impressionism style LoRA weights from
UmeAiRT/FLUX.1-dev-LoRA-Impressionism for text-to-image generation.

Available variants:
- IMPRESSIONISM: UmeAiRT/FLUX.1-dev-LoRA-Impressionism applied to FLUX.1-dev
"""

from typing import Optional

import torch
from diffusers.loaders import FluxLoraLoaderMixin
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

LORA_REPO = "UmeAiRT/FLUX.1-dev-LoRA-Impressionism"
LORA_FILENAME = "ume_classic_impressionist.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX Impressionism LoRA variants."""

    IMPRESSIONISM = "Impressionism"


class ModelLoader(ForgeModel):
    """FLUX Impressionism LoRA model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.IMPRESSIONISM: ModelConfig(
            pretrained_model_name="FLUX.1-dev-random-weights",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMPRESSIONISM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX_IMPRESSIONISM_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load FLUX.1-dev transformer with Impressionism LoRA applied.

        Uses random weights for the base model (FLUX.1-dev is a gated repo)
        and applies the publicly available Impressionism LoRA on top.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # guidance_embeds=True matches the FLUX.1-dev architecture
        self.transformer = FluxTransformer2DModel(guidance_embeds=True)
        self.transformer = self.transformer.to(dtype)
        self.transformer.eval()

        state_dict, network_alphas = FluxLoraLoaderMixin.lora_state_dict(
            LORA_REPO,
            weight_name=LORA_FILENAME,
            return_alphas=True,
        )
        FluxLoraLoaderMixin.load_lora_into_transformer(
            state_dict, network_alphas, self.transformer
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load synthetic inputs for the FLUX Impressionism LoRA transformer.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            config.pooled_projection_dim,
            dtype=dtype,
        )

        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            config.joint_attention_dim,
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

        guidance = (
            torch.full([batch_size], self.guidance_scale, dtype=dtype)
            if config.guidance_embeds
            else None
        )

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
