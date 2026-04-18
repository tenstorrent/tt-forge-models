# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Skin Diseases Classifier EfficientNetB4 model loader implementation for image classification.

This model is an EfficientNet-B4 fine-tuned for skin disease classification.
Source: https://huggingface.co/Vamsi232/Skin_Diseases_Classifier_EfficientNetB4_best
"""
import torch
import numpy as np
from datasets import load_dataset
from typing import Optional

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
    """Available Skin Diseases Classifier model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Skin Diseases Classifier EfficientNetB4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Vamsi232/Skin_Diseases_Classifier_EfficientNetB4_best",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Skin Diseases Classifier EfficientNetB4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import timm

        # The original HF model is Keras-only; use timm's EfficientNet-B4
        # with the same architecture for compile-only testing.
        model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=23)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        image = image.resize((380, 380))

        img_array = np.array(image).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        if batch_size > 1:
            img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            img_tensor = img_tensor.to(dtype_override)

        return img_tensor
