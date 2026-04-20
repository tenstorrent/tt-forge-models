# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CyberRealisticXL (cyberdelia/CyberRealisticXL) model loader implementation.

CyberRealisticXL is a Stable Diffusion XL text-to-image model specialized for
photorealistic image generation with cinematic quality, optimized for portraits,
fashion, and editorial-style scenes.

Available variants:
- CYBERREALISTIC_XL: cyberdelia/CyberRealisticXL text-to-image generation
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


REPO_ID = "cyberdelia/CyberRealisticXL"


class ModelVariant(StrEnum):
    """Available CyberRealisticXL model variants."""

    CYBERREALISTIC_XL = "CyberRealisticXL"


class ModelLoader(ForgeModel):
    """CyberRealisticXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CYBERREALISTIC_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.CYBERREALISTIC_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CyberRealisticXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CyberRealisticXL pipeline.

        Returns:
            DiffusionPipeline: The CyberRealisticXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the CyberRealisticXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
