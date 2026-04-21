# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobileNetV5 model loader implementation for image feature extraction.
"""

from typing import Optional
from dataclasses import dataclass
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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


@dataclass
class MobileNetV5Config(ModelConfig):
    """Configuration specific to MobileNetV5 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available MobileNetV5 model variants."""

    MOBILENETV5_300M_GEMMA3N = "300m_gemma3n"


class ModelLoader(ForgeModel):
    """MobileNetV5 model loader implementation for image feature extraction tasks."""

    _VARIANTS = {
        ModelVariant.MOBILENETV5_300M_GEMMA3N: MobileNetV5Config(
            pretrained_model_name="hf_hub:timm/mobilenetv5_300m.gemma3n",
            source=ModelSource.TIMM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOBILENETV5_300M_GEMMA3N

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.transform = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="MobileNetV5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=source,
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

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if self.transform is None:
            if self.model is None:
                self.load_model(dtype_override=dtype_override)
            else:
                self.transform = create_transform(
                    **resolve_data_config(self.model.pretrained_cfg, model=self.model)
                )

        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        image = image.convert("RGB")

        pixel_values = self.transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
