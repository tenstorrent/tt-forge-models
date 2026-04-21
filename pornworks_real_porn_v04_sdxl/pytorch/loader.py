# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pornworks Real Porn v0.4 SDXL (John6666/pornworks-real-porn-v04-sdxl)
model loader implementation.

Pornworks Real Porn v0.4 is a photorealism-focused text-to-image model built
on Stable Diffusion XL (SDXL).

Available variants:
- PORNWORKS_REAL_PORN_V04_SDXL: John6666/pornworks-real-porn-v04-sdxl text-to-image generation
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


REPO_ID = "John6666/pornworks-real-porn-v04-sdxl"


class ModelVariant(StrEnum):
    """Available Pornworks Real Porn v0.4 SDXL model variants."""

    PORNWORKS_REAL_PORN_V04_SDXL = "pornworks-real-porn-v04-sdxl"


class ModelLoader(ForgeModel):
    """Pornworks Real Porn v0.4 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PORNWORKS_REAL_PORN_V04_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PORNWORKS_REAL_PORN_V04_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="pornworks-real-porn-v04-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pornworks Real Porn v0.4 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Pornworks Real Porn v0.4 SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Pornworks Real Porn v0.4 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
