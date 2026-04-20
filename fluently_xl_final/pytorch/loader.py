# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fluently-XL-Final (fluently/Fluently-XL-Final) model loader implementation.

Fluently-XL-Final is a fine-tuned Stable Diffusion XL model for high-quality
text-to-image generation, based on stabilityai/stable-diffusion-xl-base-1.0.

Available variants:
- FLUENTLY_XL_FINAL: fluently/Fluently-XL-Final text-to-image generation
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline

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


REPO_ID = "fluently/Fluently-XL-Final"


class ModelVariant(StrEnum):
    """Available Fluently-XL-Final model variants."""

    FLUENTLY_XL_FINAL = "Fluently_XL_Final"


class ModelLoader(ForgeModel):
    """Fluently-XL-Final model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUENTLY_XL_FINAL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUENTLY_XL_FINAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Fluently_XL_Final",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fluently-XL-Final pipeline.

        Returns:
            DiffusionPipeline: The Fluently-XL-Final pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Fluently-XL-Final model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
