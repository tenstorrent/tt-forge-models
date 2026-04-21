# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hiera model loader implementation
"""

from typing import Optional
from dataclasses import dataclass

import timm
from transformers import AutoModelForPreTraining, AutoImageProcessor

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor
from datasets import load_dataset


@dataclass
class HieraConfig(ModelConfig):
    """Configuration specific to Hiera models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Hiera model variants."""

    TINY_224_MAE = "Tiny_224_MAE"
    BASE_PLUS_224_MAE_IN1K_FT_IN1K = "Base_Plus_224_MAE_IN1K_FT_IN1K"


class ModelLoader(ForgeModel):
    """Hiera model loader implementation."""

    _VARIANTS = {
        ModelVariant.TINY_224_MAE: HieraConfig(
            pretrained_model_name="facebook/hiera-tiny-224-mae-hf",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.BASE_PLUS_224_MAE_IN1K_FT_IN1K: HieraConfig(
            pretrained_model_name="hf_hub:timm/hiera_base_plus_224.mae_in1k_ft_in1k",
            source=ModelSource.TIMM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_224_MAE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._processor = None
        self._preprocessor = None
        self._postprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        if source == ModelSource.TIMM:
            task = ModelTask.CV_IMAGE_CLS
        else:
            task = ModelTask.CV_IMAGE_FE

        return ModelInfo(
            model="Hiera",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=task,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = AutoModelForPreTraining.from_pretrained(model_name, **kwargs)

        model.eval()
        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            if image is None:
                dataset = load_dataset("huggingface/cats-image", split="test")
                image = dataset[0]["image"]

            if self._preprocessor is None:
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=self._variant_config.pretrained_model_name,
                )
                if self.model is not None:
                    self._preprocessor.set_cached_model(self.model)

            model_for_config = self.model if self.model is not None else None

            return self._preprocessor.preprocess(
                image=image,
                dtype_override=dtype_override,
                batch_size=batch_size,
                model_for_config=model_for_config,
            )

        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        inputs = self._processor(image, return_tensors="pt")

        return inputs

    def output_postprocess(self, output, top_k=1):
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            if self._postprocessor is None:
                self._postprocessor = VisionPostprocessor(
                    model_source=source,
                    model_name=self._variant_config.pretrained_model_name,
                    model_instance=self.model,
                )

            return self._postprocessor.postprocess(
                output, top_k=top_k, return_dict=True
            )

        return output
