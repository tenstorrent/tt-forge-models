# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Xenova ViT Base Patch16-224 ONNX model loader implementation for image classification.

Xenova/vit-base-patch16-224 is an ONNX-exported variant of google/vit-base-patch16-224
distributed for use with Transformers.js.
"""

from typing import Optional

import onnx
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Xenova ViT Base Patch16-224 ONNX model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Xenova ViT Base Patch16-224 ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Xenova/vit-base-patch16-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Xenova ViT Base Patch16-224",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Xenova ViT Base Patch16-224 ONNX model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_path = hf_hub_download(pretrained_model_name, filename="onnx/model.onnx")
        model = onnx.load(model_path)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Preprocess a sample image and return model-ready input tensor."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
