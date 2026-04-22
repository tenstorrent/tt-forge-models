# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Kontext-dev style LoRA model loader implementation.

Loads the camenduru/FLUX.1-dev-ungated base pipeline (non-gated alternative
to black-forest-labs/FLUX.1-Kontext-dev) and applies style LoRA weights from
Owen777/Kontext-Style-Loras. The LoRAs target standard transformer attention
blocks and are architecture-compatible with the FLUX.1-dev backbone.

Available variants:
- KONTEXT_STYLE_3D_CHIBI: 3D Chibi style LoRA
- KONTEXT_STYLE_GHIBLI: Ghibli style LoRA
- KONTEXT_STYLE_VAN_GOGH: Van Gogh style LoRA
"""

from typing import Optional

import torch
from diffusers import FluxPipeline

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

BASE_MODEL = "camenduru/FLUX.1-dev-ungated"
LORA_REPO = "Owen777/Kontext-Style-Loras"


class ModelVariant(StrEnum):
    """Available Kontext-Style-Loras variants."""

    KONTEXT_STYLE_3D_CHIBI = "kontext_style_3d_chibi"
    KONTEXT_STYLE_GHIBLI = "kontext_style_ghibli"
    KONTEXT_STYLE_VAN_GOGH = "kontext_style_van_gogh"


_LORA_FILES = {
    ModelVariant.KONTEXT_STYLE_3D_CHIBI: "3D_Chibi_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_GHIBLI: "Ghibli_lora_weights.safetensors",
    ModelVariant.KONTEXT_STYLE_VAN_GOGH: "Van_Gogh_lora_weights.safetensors",
}


class ModelLoader(ForgeModel):
    """FLUX.1-dev style LoRA model loader with Kontext style LoRA weights."""

    _VARIANTS = {
        ModelVariant.KONTEXT_STYLE_3D_CHIBI: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.KONTEXT_STYLE_GHIBLI: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.KONTEXT_STYLE_VAN_GOGH: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KONTEXT_STYLE_3D_CHIBI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxPipeline] = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KONTEXT_STYLE_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override: Optional[torch.dtype] = None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=_LORA_FILES[self._variant],
        )
        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        prompt = "Turn this image into the Van Gogh style."
        text_inputs_clip = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipeline.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        text_inputs_t5 = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipeline.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        height_latent = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))

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

        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

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
