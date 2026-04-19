# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 FP8 model loader implementation.

Avoids accessing gated stabilityai/stable-diffusion-3.5-medium repo by using
a local transformer config and generating synthetic inputs.

Available variants:
- LARGE_FP8: sd3.5_large_fp8_scaled.safetensors
- MEDIUM_FP8: sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors
"""

from typing import Optional

import torch

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
from .src.model_utils import load_transformer, make_inputs


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 FP8 model variants."""

    LARGE_FP8 = "Large_FP8"
    MEDIUM_FP8 = "Medium_FP8"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 FP8 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_FP8: ModelConfig(
            pretrained_model_name="sd3.5_large_fp8_scaled.safetensors",
        ),
        ModelVariant.MEDIUM_FP8: ModelConfig(
            pretrained_model_name="sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_FP8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3.5 FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32
        self._transformer = load_transformer(filename, dtype=dtype)
        return self._transformer

    def load_inputs(self, dtype_override=None):
        if self._transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return make_inputs(self._transformer, dtype=dtype)
