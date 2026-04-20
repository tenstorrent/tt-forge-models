# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
briaai/FIBO model loader implementation.

FIBO is an open-source, JSON-native text-to-image diffusion model built on an
8B-parameter DiT-based flow-matching architecture with a SmolLM3-3B text
encoder and a Wan 2.2 VAE. Loaded via the custom BriaFiboPipeline.

Available variants:
- FIBO: briaai/FIBO text-to-image generation
"""

from typing import Optional

import torch
from diffusers import BriaFiboPipeline

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


REPO_ID = "briaai/FIBO"


class ModelVariant(StrEnum):
    """Available FIBO model variants."""

    FIBO = "FIBO"


class ModelLoader(ForgeModel):
    """briaai/FIBO model loader implementation."""

    _VARIANTS = {
        ModelVariant.FIBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FIBO

    DEFAULT_PROMPT = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FIBO pipeline.

        Returns:
            BriaFiboPipeline: The FIBO pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = BriaFiboPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the FIBO model.

        Returns:
            list: A list of sample text prompts.
        """
        return [self.DEFAULT_PROMPT] * batch_size
