# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Xenova DINO ViT-S/16 model loader implementation for image feature extraction.

Xenova/dino-vits16 is a Transformers.js-targeted repackaging of
facebook/dino-vits16 (Vision Transformer Small patch-16 pretrained with DINO).
"""
import torch
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Xenova DINO ViT-S/16 model variants."""

    SMALL_16 = "Small_16"


class ModelLoader(ForgeModel):
    """Xenova DINO ViT-S/16 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.SMALL_16: ModelConfig(
            pretrained_model_name="Xenova/dino-vits16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="XenovaDinoViTS16",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ViTModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self.processor is None:
            self._load_processor()

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
