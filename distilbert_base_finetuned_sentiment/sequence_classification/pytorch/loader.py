# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT Base Finetuned Sentiment (lyrisha/distilbert-base-finetuned-sentiment) model loader implementation for sequence classification.
"""

from transformers import DistilBertForSequenceClassification, AutoTokenizer
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
    """Available DistilBERT Base Finetuned Sentiment model variants for sequence classification."""

    DISTILBERT_BASE_FINETUNED_SENTIMENT = "lyrisha/distilbert-base-finetuned-sentiment"


class ModelLoader(ForgeModel):
    """DistilBERT Base Finetuned Sentiment model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_FINETUNED_SENTIMENT: LLMModelConfig(
            pretrained_model_name="lyrisha/distilbert-base-finetuned-sentiment",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_FINETUNED_SENTIMENT

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "I absolutely loved this movie, it was fantastic!"
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DistilBERT_Base_Finetuned_Sentiment",
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

        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
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

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sentiment classification."""
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_category = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted sentiment: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
