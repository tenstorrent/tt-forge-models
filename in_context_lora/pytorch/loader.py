# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-Context LoRA for FLUX.1-dev model loader implementation.

Loads the camenduru/FLUX.1-dev-ungated base pipeline and applies task-specific
In-Context LoRA weights from ali-vilab/In-Context-LoRA for multi-task
text-to-image generation with customizable intrinsic relationships.

Available variants:
- COUPLE_PROFILE: Couple profile design LoRA
- FILM_STORYBOARD: Film storyboard LoRA
- FONT_DESIGN: Font design LoRA
- HOME_DECORATION: Home decoration LoRA
- PORTRAIT_ILLUSTRATION: Portrait illustration LoRA
- PORTRAIT_PHOTOGRAPHY: Portrait photography LoRA
- PPT_TEMPLATES: PPT template LoRA
- SANDSTORM_VISUAL_EFFECT: Sandstorm visual effect LoRA
- SPARKLERS_VISUAL_EFFECT: Sparklers visual effect LoRA
- VISUAL_IDENTITY_DESIGN: Visual identity design LoRA
"""

import os
from typing import Any, Optional

import torch
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

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
LORA_REPO = "ali-vilab/In-Context-LoRA"


class ModelVariant(StrEnum):
    """Available In-Context LoRA task variants."""

    COUPLE_PROFILE = "couple-profile"
    FILM_STORYBOARD = "film-storyboard"
    FONT_DESIGN = "font-design"
    HOME_DECORATION = "home-decoration"
    PORTRAIT_ILLUSTRATION = "portrait-illustration"
    PORTRAIT_PHOTOGRAPHY = "portrait-photography"
    PPT_TEMPLATES = "ppt-templates"
    SANDSTORM_VISUAL_EFFECT = "sandstorm-visual-effect"
    SPARKLERS_VISUAL_EFFECT = "sparklers-visual-effect"
    VISUAL_IDENTITY_DESIGN = "visual-identity-design"


_LORA_FILES = {
    ModelVariant.COUPLE_PROFILE: "couple-profile.safetensors",
    ModelVariant.FILM_STORYBOARD: "film-storyboard.safetensors",
    ModelVariant.FONT_DESIGN: "font-design.safetensors",
    ModelVariant.HOME_DECORATION: "home-decoration.safetensors",
    ModelVariant.PORTRAIT_ILLUSTRATION: "portrait-illustration.safetensors",
    ModelVariant.PORTRAIT_PHOTOGRAPHY: "portrait-photography.safetensors",
    ModelVariant.PPT_TEMPLATES: "ppt-templates.safetensors",
    ModelVariant.SANDSTORM_VISUAL_EFFECT: "sandstorm-visual-effect.safetensors",
    ModelVariant.SPARKLERS_VISUAL_EFFECT: "sparklers-visual-effect.safetensors",
    ModelVariant.VISUAL_IDENTITY_DESIGN: "visual-identity-design.safetensors",
}


class ModelLoader(ForgeModel):
    """In-Context LoRA for FLUX.1-dev model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=BASE_MODEL)
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.COUPLE_PROFILE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxPipeline] = None
        self._transformer: Optional[FluxTransformer2DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="IN_CONTEXT_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = FluxTransformer2DModel.load_config(
                self._variant_config.pretrained_model_name,
                subfolder="transformer",
            )
            self._transformer = FluxTransformer2DModel.from_config(config).to(dtype)
        else:
            self.pipeline = FluxPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                **kwargs,
            )
            lora_file = _LORA_FILES[self._variant]
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=lora_file,
            )
            self._transformer = self.pipeline.transformer

        return self._transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        height = 128
        width = 128
        max_sequence_length = 256
        num_images_per_prompt = 1
        guidance_scale = 3.5
        vae_scale_factor = 8

        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            h_packed,
            2,
            w_packed,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            h_packed * w_packed,
            num_channels_latents * 4,
        )

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

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        guidance = (
            torch.full(
                [batch_size * num_images_per_prompt], guidance_scale, dtype=dtype
            )
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
