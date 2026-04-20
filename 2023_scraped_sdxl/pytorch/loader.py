# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
2023 Scraped SDXL (MLbackup/2023_Scraped_SDXL) model loader implementation.

2023 Scraped SDXL is an anime-oriented text-to-image model fine-tuned from
Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0).

Available variants:
- SCRAPED_SDXL_2023: MLbackup/2023_Scraped_SDXL text-to-image generation
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


REPO_ID = "MLbackup/2023_Scraped_SDXL"


class ModelVariant(StrEnum):
    """Available 2023 Scraped SDXL model variants."""

    SCRAPED_SDXL_2023 = "2023_Scraped_SDXL"


class ModelLoader(ForgeModel):
    """2023 Scraped SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.SCRAPED_SDXL_2023: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SCRAPED_SDXL_2023

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="2023_Scraped_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the 2023 Scraped SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The 2023 Scraped SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the 2023 Scraped SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
