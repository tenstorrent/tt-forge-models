# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Docling Layout Egret model loader implementation for document layout detection.
"""
import torch
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
    """Available Docling Layout Egret model variants."""

    DOCLING_LAYOUT_EGRET_LARGE = "docling_layout_egret_large"


class ModelLoader(ForgeModel):
    """Docling Layout Egret model loader for document layout detection."""

    _VARIANTS = {
        ModelVariant.DOCLING_LAYOUT_EGRET_LARGE: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-egret-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOCLING_LAYOUT_EGRET_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Docling Layout Egret",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import RTDetrImageProcessor

        self.processor = RTDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import DFineForObjectDetection

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DFineForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        import requests
        from PIL import Image

        if self.processor is None:
            self._load_processor()

        url = "https://huggingface.co/spaces/ds4sd/SmolDocling-256M-Demo/resolve/main/example_images/annual_rep_14.png"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
