# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
zai-org/CogView4-6B model loader implementation.

CogView4-6B is a 6B parameter text-to-image generation model built on the
GLM-4-9B base model, supporting Chinese and English prompts.

Available variants:
- COGVIEW4_6B: zai-org/CogView4-6B text-to-image generation
"""

from typing import Optional

import torch
from diffusers import CogView4Pipeline

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
    """Available CogView4 model variants."""

    COGVIEW4_6B = "CogView4-6B"


class ModelLoader(ForgeModel):
    """CogView4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.COGVIEW4_6B: ModelConfig(
            pretrained_model_name="zai-org/CogView4-6B",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.COGVIEW4_6B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CogView4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CogView4 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            CogView4Pipeline: The CogView4 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = CogView4Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the CogView4 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A vibrant cherry red sports car sits proudly under the gleaming sun, "
            "its polished exterior smooth and flawless."
        ] * batch_size
