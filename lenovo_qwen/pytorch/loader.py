#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lenovo Qwen LoRA model loader implementation.

Loads the Qwen/Qwen-Image base diffusion pipeline and applies the
Danrisi/Lenovo_Qwen LoRA weights for realistic amateur-style candid
photography with controllable indoor/outdoor and exposure attributes.

Available variants:
- LENOVO_QWEN: Lenovo Qwen LoRA on Qwen-Image
"""

from typing import Any, Optional

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

BASE_MODEL = "Qwen/Qwen-Image"
LORA_REPO = "Danrisi/Lenovo_Qwen"
LORA_WEIGHT_NAME = "lenovo.safetensors"


class ModelVariant(StrEnum):
    """Available Lenovo Qwen model variants."""

    LENOVO_QWEN = "Lenovo_Qwen"


class ModelLoader(ForgeModel):
    """Lenovo Qwen LoRA model loader."""

    _VARIANTS = {
        ModelVariant.LENOVO_QWEN: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LENOVO_QWEN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LENOVO_QWEN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image pipeline with Lenovo Qwen LoRA weights.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for realistic candid photo generation.

        Returns:
            dict with prompt and negative_prompt keys.
        """
        prompt = (
            "overexposed indoor scene, raw unedited amateurish candid shot of "
            "a woman standing by a window in a cluttered apartment"
        )
        negative_prompt = (
            "blurry, worst quality, low quality, jpeg artifacts, "
            "professional studio lighting, overly polished"
        )

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
