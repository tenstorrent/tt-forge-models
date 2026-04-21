# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compassmix XL Lightning model loader implementation.

Compassmix XL Lightning is a Stable Diffusion XL fine-tune distributed by
frosting.ai, tuned for fast text-to-image generation (8 recommended steps).

Available variants:
- COMPASSMIX_XL_LIGHTNING: frosting-ai/compassmix-xl-lightning text-to-image generation
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
    """Available Compassmix XL Lightning model variants."""

    COMPASSMIX_XL_LIGHTNING = "compassmix-xl-lightning"


class ModelLoader(ForgeModel):
    """Compassmix XL Lightning model loader implementation."""

    _VARIANTS = {
        ModelVariant.COMPASSMIX_XL_LIGHTNING: ModelConfig(
            pretrained_model_name="frosting-ai/compassmix-xl-lightning",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.COMPASSMIX_XL_LIGHTNING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Compassmix_XL_Lightning",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Compassmix XL Lightning pipeline.

        Returns:
            StableDiffusionXLPipeline: The Compassmix XL Lightning pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Compassmix XL Lightning model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic photo of a lighthouse on a cliff during a storm, dramatic lighting, high detail"
        ] * batch_size
