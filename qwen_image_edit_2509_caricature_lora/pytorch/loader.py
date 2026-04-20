#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2509-Caricature-LoRA model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2509 base pipeline and applies LoRA weights
from drbaph/Qwen-Image-Edit-2509-Caricature-LoRA to transform input images
into sketched caricature art with exaggerated features.

Available variants:
- V1_1: Default LoRA weights (qwen-edit-2509-caricature_v1.1.safetensors)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline  # type: ignore[import]
from PIL import Image  # type: ignore[import]

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "drbaph/Qwen-Image-Edit-2509-Caricature-LoRA"
LORA_WEIGHT_NAME = "qwen-edit-2509-caricature_v1.1.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2509-Caricature-LoRA variants."""

    V1_1 = "v1.1"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2509-Caricature-LoRA model loader."""

    _VARIANTS = {
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_2509_CARICATURE_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen Image Edit pipeline with Caricature LoRA weights applied.

        Returns:
            QwenImageEditPlusPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for caricature-style image editing.

        Returns:
            dict with prompt and image keys.
        """
        if prompt is None:
            prompt = "create a caricature of this image"

        image = Image.new("RGB", (1024, 1024), color=(128, 128, 200))

        return {
            "prompt": prompt,
            "image": image,
        }
