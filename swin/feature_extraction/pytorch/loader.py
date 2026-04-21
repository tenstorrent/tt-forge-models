# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin feature extraction model loader.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


@dataclass
class SwinFEConfig(ModelConfig):
    """Configuration specific to Swin feature extraction models."""

    source: ModelSource = ModelSource.HUGGING_FACE


class ModelVariant(StrEnum):
    """Available Swin feature extraction model variants."""

    TINY_RANDOM = "TinyRandom"
    TINY_RANDOM_PATCH4_WINDOW7_224 = "Tiny_Random_Patch4_Window7_224"


class ModelLoader(ForgeModel):
    """Swin feature extraction model loader."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: SwinFEConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-SwinModel",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SWIN_TINY_PATCH4_WINDOW7_224_CTRANSPATH: SwinFEConfig(
            pretrained_model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            source=ModelSource.TIMM,
        ),
        ModelVariant.TINY_RANDOM_PATCH4_WINDOW7_224: ModelConfig(
            pretrained_model_name="yujiepan/tiny-random-swin-patch4-window7-224",
        ),
        ModelVariant.TINY_RANDOM_PATCH4_WINDOW7_224: ModelConfig(
            pretrained_model_name="yujiepan/tiny-random-swin-patch4-window7-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="Swin",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=source,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            import timm

            model = timm.create_model(
                pretrained_model_name,
                embed_layer=ConvStem,
                pretrained=True,
            )
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)

        model.eval()

        if dtype_override is not None and source == ModelSource.TIMM:
            model = model.to(dtype_override)

        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        source = self._variant_config.source

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        if source == ModelSource.TIMM:
            from timm.data import create_transform, resolve_data_config

            model_for_config = self._cached_model
            if model_for_config is None:
                model_for_config = self.load_model(dtype_override=dtype_override)

            data_config = resolve_data_config({}, model=model_for_config)
            transforms = create_transform(**data_config, is_training=False)
            inputs = transforms(image).unsqueeze(0)
            inputs = inputs.repeat_interleave(batch_size, dim=0)

            if dtype_override is not None:
                inputs = inputs.to(dtype_override)
            return inputs

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
