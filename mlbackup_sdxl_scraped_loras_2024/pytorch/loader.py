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
from diffusers import StableDiffusionXLPipeline

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


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_REPO = "MLbackup/SDXL_Scraped_Loras_2024"


class ModelVariant(StrEnum):
    """Available MLbackup SDXL_Scraped_Loras_2024 variants."""

    DUSK_XL_TAROTCARD = "DUSK_XL_TAROTCARD"


_LORA_WEIGHT_NAMES = {
    ModelVariant.DUSK_XL_TAROTCARD: "DUSK_XL_TAROTCARD_dadapt_cos_5e4.safetensors",
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
        self.pipeline: Optional[StableDiffusionXLPipeline] = None

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
        """Load the SDXL base pipeline with the archived LoRA weights applied.

        Returns:
            StableDiffusionXLPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=_LORA_WEIGHT_NAMES[self._variant],
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "an ornate tarot card illustration of a cloaked traveler under a crescent moon, mystical symbols"
        ] * batch_size
