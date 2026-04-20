# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RETFound MAE model loader implementation for retinal image feature extraction.

RETFound_MAE is a ViT-Large/16/224 foundation model pretrained with the Masked
Autoencoder (MAE) self-supervised method on retinal images, distributed through
timm via bitfount/RETFound_MAE.
"""
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from typing import Optional
from datasets import load_dataset

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
    """Available RETFound MAE model variants."""

    VIT_LARGE_PATCH16_224 = "ViT_Large_Patch16_224"


class ModelLoader(ForgeModel):
    """RETFound MAE model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_LARGE_PATCH16_224: ModelConfig(
            pretrained_model_name="hf-hub:bitfount/RETFound_MAE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_LARGE_PATCH16_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RETFound-MAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        self.model = model
        self.transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform is None:
            if self.model is None:
                self.load_model()
            else:
                self.transform = create_transform(
                    **resolve_data_config(self.model.pretrained_cfg, model=self.model)
                )

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        pixel_values = self.transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
