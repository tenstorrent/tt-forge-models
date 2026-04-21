# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HeBERT model loader implementation for Hebrew sentiment analysis.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available HeBERT model variants for sentiment analysis."""

    AVICHR_HEBERT_SENTIMENT_ANALYSIS = "avichr_heBERT_sentiment_analysis"


class ModelLoader(ForgeModel):
    """HeBERT model loader implementation for Hebrew sentiment analysis."""

    _VARIANTS = {
        ModelVariant.AVICHR_HEBERT_SENTIMENT_ANALYSIS: LLMModelConfig(
            pretrained_model_name="avichr/heBERT_sentiment_analysis",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AVICHR_HEBERT_SENTIMENT_ANALYSIS

    # Sample Hebrew text for sentiment analysis ("Coffee is delicious")
    sample_text = "קפה זה טעים"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="HeBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Sentiment: {self.model.config.id2label[predicted_value]}")
