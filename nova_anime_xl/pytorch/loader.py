# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nova Anime XL (John6666/nova-anime-xl-il-v80-sdxl) model loader implementation.

Nova Anime XL is an SDXL-based model fine-tuned for anime and illustration
style image generation, merged from Illustrious-XL-v2.0 and noobai-XL-1.1.

Available variants:
- NOVA_ANIME_XL: John6666/nova-anime-xl-il-v80-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

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


REPO_ID = "John6666/nova-anime-xl-il-v80-sdxl"


class ModelVariant(StrEnum):
    """Available Nova Anime XL model variants."""

    NOVA_ANIME_XL = "Nova_Anime_XL"


class ModelLoader(ForgeModel):
    """Nova Anime XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOVA_ANIME_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NOVA_ANIME_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Nova_Anime_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Nova Anime XL pipeline and return its UNet.

        Returns:
            UNet2DConditionModel: The UNet component of the pipeline.
        """
        # Load pipeline in float32 for CPU-compatible preprocessing
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
            **kwargs,
        )
        # Ensure all components are float32 for CPU preprocessing
        self.pipeline.to("cpu", torch.float32)

        unet = self.pipeline.unet
        unet.eval()
        for param in unet.parameters():
            param.requires_grad = False

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return preprocessed UNet inputs.

        Returns:
            list: [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = "A beautiful anime girl in a fantasy landscape with colorful flowers"
        device = "cpu"
        dtype = dtype_override if dtype_override is not None else torch.float32

        height = self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        width = self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            device=device,
            num_images_per_prompt=1,
        )

        timesteps, _ = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps=50,
            device=device,
        )

        num_channels_latents = self.pipeline.unet.config.in_channels
        torch.manual_seed(42)
        latents = torch.randn(
            (
                batch_size,
                num_channels_latents,
                height // self.pipeline.vae_scale_factor,
                width // self.pipeline.vae_scale_factor,
            ),
            device=device,
        )
        latents = latents * self.pipeline.scheduler.init_noise_sigma

        add_text_embeds = pooled_prompt_embeds
        if self.pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = (
                self.pipeline.text_encoder_2.config.projection_dim
            )
        add_time_ids = self.pipeline._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.pipeline.scheduler.scale_model_input(
            latent_model_input, timesteps[0]
        )

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
