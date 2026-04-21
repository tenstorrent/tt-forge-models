# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Korean Sentiment model loader implementation for sequence classification.

Loads the matthewburke/korean_sentiment model, a KcELECTRA-base fine-tune
for binary Korean sentiment classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available Korean Sentiment model variants."""

    MATTHEWBURKE_KOREAN_SENTIMENT = "matthewburke_korean_sentiment"


class ModelLoader(ForgeModel):
    """Korean Sentiment model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.MATTHEWBURKE_KOREAN_SENTIMENT: LLMModelConfig(
            pretrained_model_name="matthewburke/korean_sentiment",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MATTHEWBURKE_KOREAN_SENTIMENT

    sample_text = "영화 재밌다."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Korean_Sentiment",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
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

    def decode_output(self, co_out):
        predicted_class_id = co_out[0].argmax(-1).item()
        if self.model is not None and hasattr(self.model.config, "id2label"):
            predicted_label = self.model.config.id2label[predicted_class_id]
            print(f"Predicted Sentiment: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
