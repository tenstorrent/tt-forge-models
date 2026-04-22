# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLM2CLIP model loader implementation for image feature extraction.
"""

import sys
import types
from typing import Optional

import torch
import torch.nn as nn

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


def _stub_transformers_onnx():
    """Stub out transformers.onnx which was removed in transformers 5.x.

    The LLM2CLIP remote code imports OnnxConfig from transformers.onnx at
    module level, but only uses it for ONNX export (not inference).
    """
    if "transformers.onnx" not in sys.modules:
        stub = types.ModuleType("transformers.onnx")
        stub.OnnxConfig = object
        sys.modules["transformers.onnx"] = stub
        import transformers

        transformers.onnx = stub


class LLM2CLIPVisionWrapper(nn.Module):
    """Wrap LLM2CLIPModel to expose only the image feature extraction path.

    LLM2CLIPModel.forward() requires text inputs (self.text_model) which are
    not part of this checkpoint. Use get_image_features() instead.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values=pixel_values)


class ModelVariant(StrEnum):
    """Available LLM2CLIP model variants."""

    OPENAI_L_14_336 = "Openai-L-14-336"


class ModelLoader(ForgeModel):
    """LLM2CLIP model loader implementation for image feature extraction."""

    _VARIANTS = {
        ModelVariant.OPENAI_L_14_336: ModelConfig(
            pretrained_model_name="microsoft/LLM2CLIP-Openai-L-14-336",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENAI_L_14_336

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LLM2CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import CLIPImageProcessor

        self._processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        # transformers.onnx was removed in transformers 5.x but the LLM2CLIP
        # remote code imports OnnxConfig from it at module level.
        _stub_transformers_onnx()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        # LLM2CLIPModel.forward() requires text inputs via self.text_model, but
        # this checkpoint only has vision weights. Wrap to use get_image_features.
        return LLM2CLIPVisionWrapper(model)

    def load_inputs(self, dtype_override=None):
        # datasets._dill checks for spacy.Language, but the spacy/ directory in
        # this repo creates a namespace package without it, causing AttributeError.
        # Temporarily hide the stub so datasets skips the spacy check.
        spacy_stub = sys.modules.pop("spacy", None)
        spacy_is_stub = spacy_stub is not None and not hasattr(spacy_stub, "Language")
        if not spacy_is_stub and spacy_stub is not None:
            sys.modules["spacy"] = spacy_stub
            spacy_stub = None

        try:
            from datasets import load_dataset

            if self._processor is None:
                self._load_processor(dtype_override=dtype_override)

            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

            inputs = self._processor(images=image, return_tensors="pt")

            if dtype_override is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

            return inputs
        finally:
            if spacy_stub is not None:
                sys.modules["spacy"] = spacy_stub
