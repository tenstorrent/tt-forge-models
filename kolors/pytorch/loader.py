# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kolors (Kwai-Kolors/Kolors-diffusers) model loader implementation.

Kolors is a large-scale text-to-image latent diffusion model developed by Kuaishou.
It is trained on billions of text-image pairs and supports both Chinese and English
prompts with strong visual quality and text rendering capabilities.

Available variants:
- KOLORS: Kwai-Kolors/Kolors-diffusers text-to-image generation
"""

from typing import Optional

import torch
from diffusers import KolorsPipeline

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


REPO_ID = "Kwai-Kolors/Kolors-diffusers"


class ModelVariant(StrEnum):
    """Available Kolors model variants."""

    KOLORS = "Kolors"


class ModelLoader(ForgeModel):
    """Kolors model loader implementation."""

    _VARIANTS = {
        ModelVariant.KOLORS: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.KOLORS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kolors",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kolors pipeline.

        Returns:
            KolorsPipeline: The Kolors pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = KolorsPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Kolors model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A photo of a ladybug, macro, zoom, high quality, cinematic"
        ] * batch_size
