#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 360 Diffusion LoRA model loader implementation.

Loads the Qwen-Image base pipeline and applies the 360-degree equirectangular
panorama LoRA weights from ProGamerGov/qwen-360-diffusion for text-to-image
generation of 360-degree panoramic images.

Available variants:
- INT8_V1: int8-bf16 v1 LoRA on Qwen/Qwen-Image (default)
- INT4_V1: int4-bf16 v1 LoRA on Qwen/Qwen-Image
- INT4_V1B: int4-bf16 v1-b LoRA on Qwen/Qwen-Image
- INT8_V2_2512: int8-bf16 v2 LoRA on Qwen/Qwen-Image-2512
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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

BASE_MODEL_QWEN_IMAGE = "Qwen/Qwen-Image"
BASE_MODEL_QWEN_IMAGE_2512 = "Qwen/Qwen-Image-2512"
LORA_REPO = "ProGamerGov/qwen-360-diffusion"

LORA_INT8_V1 = "qwen-360-diffusion-int8-bf16-v1.safetensors"
LORA_INT4_V1 = "qwen-360-diffusion-int4-bf16-v1.safetensors"
LORA_INT4_V1B = "qwen-360-diffusion-int4-bf16-v1-b.safetensors"
LORA_INT8_V2_2512 = "qwen-360-diffusion-2512-int8-bf16-v2.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen 360 Diffusion LoRA variants."""

    INT8_V1 = "int8_v1"
    INT4_V1 = "int4_v1"
    INT4_V1B = "int4_v1b"
    INT8_V2_2512 = "int8_v2_2512"


_LORA_FILES = {
    ModelVariant.INT8_V1: LORA_INT8_V1,
    ModelVariant.INT4_V1: LORA_INT4_V1,
    ModelVariant.INT4_V1B: LORA_INT4_V1B,
    ModelVariant.INT8_V2_2512: LORA_INT8_V2_2512,
}


class ModelLoader(ForgeModel):
    """Qwen 360 Diffusion LoRA model loader."""

    _VARIANTS = {
        ModelVariant.INT8_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1B: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT8_V2_2512: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE_2512,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INT8_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_360_DIFFUSION",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image pipeline with 360 diffusion LoRA weights applied.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for 360-degree panorama image generation.

        Returns:
            dict with prompt and generation parameters.
        """
        if prompt is None:
            prompt = (
                "equirectangular 360 panorama of a mountain landscape at sunset, "
                "photography, 4K"
            )

        return {
            "prompt": prompt,
            "width": 2048,
            "height": 1024,
        }
