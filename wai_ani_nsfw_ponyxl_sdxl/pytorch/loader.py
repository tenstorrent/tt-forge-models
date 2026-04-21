# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAI ANI NSFW PonyXL v140 (John6666/wai-ani-nsfw-ponyxl-v140-sdxl) model loader implementation.

WAI ANI NSFW PonyXL is a Stable Diffusion XL checkpoint fine-tuned on the
Pony Diffusion base for anime-style text-to-image generation.

Available variants:
- WAI_ANI_NSFW_PONYXL_V140: John6666/wai-ani-nsfw-ponyxl-v140-sdxl text-to-image generation
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


REPO_ID = "John6666/wai-ani-nsfw-ponyxl-v140-sdxl"


class ModelVariant(StrEnum):
    """Available WAI ANI NSFW PonyXL model variants."""

    WAI_ANI_NSFW_PONYXL_V140 = "WAI_ANI_NSFW_PonyXL_v140"


class ModelLoader(ForgeModel):
    """WAI ANI NSFW PonyXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAI_ANI_NSFW_PONYXL_V140: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAI_ANI_NSFW_PONYXL_V140

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAI_ANI_NSFW_PonyXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WAI ANI NSFW PonyXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The WAI ANI NSFW PonyXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the WAI ANI NSFW PonyXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
