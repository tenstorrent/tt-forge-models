# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NAICSKit v1 model loader implementation for sequence classification.

Loads the ndixon104/naicskit-v1 model, a T5-small based binary
sequence classifier (NEGATIVE/POSITIVE).
"""

from transformers import AutoTokenizer, T5ForSequenceClassification
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available NAICSKit v1 model variants."""

    NAICSKIT_V1 = "naicskit_v1"


class ModelLoader(ForgeModel):
    """NAICSKit v1 model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.NAICSKIT_V1: LLMModelConfig(
            pretrained_model_name="ndixon104/naicskit-v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NAICSKIT_V1

    sample_text = "Offices of physicians providing general medical care."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NAICSKit_v1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax(-1).item()
        model = framework_model if framework_model is not None else self._cached_model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_label = model.config.id2label[predicted_class_id]
            print(f"Predicted Label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
