# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa Large Financial News Sentiment EN (Jean-Baptiste/roberta-large-financial-news-sentiment-en) model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available RoBERTa Large Financial News Sentiment EN model variants for sequence classification."""

    ROBERTA_LARGE_FINANCIAL_NEWS_SENTIMENT_EN = (
        "Jean-Baptiste/roberta-large-financial-news-sentiment-en"
    )


class ModelLoader(ForgeModel):
    """RoBERTa Large Financial News Sentiment EN model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.ROBERTA_LARGE_FINANCIAL_NEWS_SENTIMENT_EN: LLMModelConfig(
            pretrained_model_name="Jean-Baptiste/roberta-large-financial-news-sentiment-en",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBERTA_LARGE_FINANCIAL_NEWS_SENTIMENT_EN

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "Melcor REIT (TSX: MR.UN) today announced results for the third quarter "
            "ended September 30, 2022. Revenue was stable in the quarter and "
            "year-to-date. Net operating income was down 3% in the quarter at "
            "$11.61 million due to the timing of operating expenses and inflated "
            "costs including utilities like gas/heat and power."
        )
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="RoBERTa_Large_Financial_News_Sentiment_EN",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RoBERTa Large Financial News Sentiment EN model for sequence classification from Hugging Face."""
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
        """Prepare sample input for RoBERTa Large Financial News Sentiment EN sequence classification."""
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

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for financial news sentiment classification."""
        predicted_class_id = co_out[0].argmax(-1).item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_sentiment = model.config.id2label[predicted_class_id]
            print(f"Predicted Sentiment: {predicted_sentiment}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
