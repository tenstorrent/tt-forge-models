# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup/Scraped_2025_Loras model loader implementation.

Loads a Stable Diffusion XL family base pipeline and applies a LoRA adapter
from MLbackup/Scraped_2025_Loras for text-to-image generation. The repo is a
collection of LoRAs targeting several SDXL-lineage base models (Illustrious XL,
Pony XL, NoobAI XL, and vanilla SDXL); each variant below pairs one LoRA
safetensors file with its matching base model.

Available variants:
- ARCANE_STYLE_ILLUSTRIOUS: Arcane style LoRA on Illustrious XL
- ARCANE_STYLE_PONYXL: Arcane style LoRA on SDXL base (Pony XL-compatible)
- NOOBAI_32K_UHD_AESTHETIC: 32k UHD aesthetic LoRA on NoobAI XL 1.1
- MICRO_CUBE_WORLDS_SDXL: Micro Cube Worlds LoRA on SDXL base 1.0
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image
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


LORA_REPO = "MLbackup/Scraped_2025_Loras"

PROMPT = "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."


class ModelVariant(StrEnum):
    """Available MLbackup Scraped_2025_Loras variants."""

    ARCANE_STYLE_ILLUSTRIOUS = "Arcane_Style_IllustriousXL"
    ARCANE_STYLE_PONYXL = "Arcane_Style_PonyXL"
    NOOBAI_32K_UHD_AESTHETIC = "NoobAI_32k_UHD_Aesthetic"
    MICRO_CUBE_WORLDS_SDXL = "Micro_Cube_Worlds_SDXL_BASE"


_LORA_FILES = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "Arcane_Style_IllustriousXL.safetensors",
    ModelVariant.ARCANE_STYLE_PONYXL: "Arcane_Style_PonyXL.safetensors",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "NoobAI_32k_UHD_Aesthetic.safetensors",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "Micro_Cube_Worlds_SDXL_BASE.safetensors",
}

# AstraliteHeart/pony-diffusion-v6 is a single-file SD1.5 checkpoint without
# a diffusers model_index.json; use the standard SDXL base instead since the
# LoRA targets SDXL architecture.
_BASE_MODELS = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "OnomaAIResearch/Illustrious-xl-early-release-v0",
    ModelVariant.ARCANE_STYLE_PONYXL: "stabilityai/stable-diffusion-xl-base-1.0",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "Laxhar/noobai-XL-1.1",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "stabilityai/stable-diffusion-xl-base-1.0",
}


class ModelLoader(ForgeModel):
    """MLbackup Scraped_2025_Loras model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=_BASE_MODELS[variant])
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.ARCANE_STYLE_ILLUSTRIOUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Scraped_2025_Loras",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load an SDXL-family pipeline, apply the selected LoRA, and return the UNet.

        Returns:
            torch.nn.Module: The UNet model with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample UNet inputs for the selected LoRA variant.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        pipe = self.pipeline
        device = "cpu"
        height = pipe.default_sample_size * pipe.vae_scale_factor
        width = pipe.default_sample_size * pipe.vae_scale_factor

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=PROMPT,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            device=device,
            num_images_per_prompt=1,
        )

        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps=50,
            device=device,
        )

        num_channels_latents = pipe.unet.config.in_channels
        torch.manual_seed(42)
        latents = torch.randn(
            (
                batch_size,
                num_channels_latents,
                height // pipe.vae_scale_factor,
                width // pipe.vae_scale_factor,
            ),
            device=device,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, timesteps[0]
        )

        if dtype_override is not None:
            latent_model_input = latent_model_input.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            },
        }
