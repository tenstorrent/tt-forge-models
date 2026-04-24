# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup/SDXL_Scraped_Loras_2024 model loader implementation.

Loads stabilityai/stable-diffusion-xl-base-1.0 and applies a LoRA weight
archived in MLbackup/SDXL_Scraped_Loras_2024 for stylized text-to-image
generation.

Available variants:
- DUSK_XL_TAROTCARD: Dusk tarot-card style SDXL LoRA
"""

from typing import Optional

import torch

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_REPO = "MLbackup/SDXL_Scraped_Loras_2024"


class ModelVariant(StrEnum):
    """Available MLbackup SDXL_Scraped_Loras_2024 variants."""

    DUSK_XL_TAROTCARD = "DUSK_XL_TAROTCARD"


_LORA_WEIGHT_NAMES = {
    ModelVariant.DUSK_XL_TAROTCARD: "DUSK_XL_TAROTCARD_dadapt_cos_5e4.safetensors",
}

_PROMPTS = {
    ModelVariant.DUSK_XL_TAROTCARD: "an ornate tarot card illustration of a cloaked traveler under a crescent moon, mystical symbols",
}


class ModelLoader(ForgeModel):
    """MLbackup/SDXL_Scraped_Loras_2024 LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.DUSK_XL_TAROTCARD: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DUSK_XL_TAROTCARD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MLbackup_SDXL_Scraped_Loras_2024",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SDXL base pipeline with LoRA weights fused and return the UNet.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name,
            LORA_REPO,
            _LORA_WEIGHT_NAMES[self._variant],
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = _PROMPTS[self._variant]
        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, prompt)

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
