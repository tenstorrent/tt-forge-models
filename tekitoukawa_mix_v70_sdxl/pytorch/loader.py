# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tekitoukawa Mix v70 SDXL (John6666/tekitoukawa-mix-v70-sdxl) model loader implementation.

Tekitoukawa Mix v70 SDXL is an anime-focused Stable Diffusion XL fine-tune of
Illustrious XL optimized for cute anime-style illustration generation.

Available variants:
- TEKITOUKAWA_MIX_V70_SDXL: John6666/tekitoukawa-mix-v70-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "John6666/tekitoukawa-mix-v70-sdxl"


class ModelVariant(StrEnum):
    """Available Tekitoukawa Mix v70 SDXL model variants."""

    TEKITOUKAWA_MIX_V70_SDXL = "tekitoukawa-mix-v70-sdxl"


class ModelLoader(ForgeModel):
    """Tekitoukawa Mix v70 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.TEKITOUKAWA_MIX_V70_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TEKITOUKAWA_MIX_V70_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="tekitoukawa-mix-v70-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Tekitoukawa Mix v70 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Tekitoukawa Mix v70 SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Tekitoukawa Mix v70 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
