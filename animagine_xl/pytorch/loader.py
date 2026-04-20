# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Animagine XL model loader implementation.

Animagine XL is an anime-focused text-to-image model family based on Stable Diffusion XL,
fine-tuned for high-quality anime-style image generation.

Available variants:
- ANIMAGINE_XL_3_1: votepurchase/animagine-xl-3.1 text-to-image generation
- ANIMAGINE_XL_4_0_V4OPT: John6666/animagine-xl-40-v4opt-sdxl text-to-image generation
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


class ModelVariant(StrEnum):
    """Available Animagine XL model variants."""

    ANIMAGINE_XL_3_1 = "Animagine_XL_3_1"
    ANIMAGINE_XL_4_0_V4OPT = "Animagine_XL_4_0_v4opt"


class ModelLoader(ForgeModel):
    """Animagine XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANIMAGINE_XL_3_1: ModelConfig(
            pretrained_model_name="votepurchase/animagine-xl-3.1",
        ),
        ModelVariant.ANIMAGINE_XL_4_0_V4OPT: ModelConfig(
            pretrained_model_name="John6666/animagine-xl-40-v4opt-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMAGINE_XL_3_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Animagine_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Animagine XL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Animagine XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Animagine XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck, masterpiece, best quality, very aesthetic"
        ] * batch_size
