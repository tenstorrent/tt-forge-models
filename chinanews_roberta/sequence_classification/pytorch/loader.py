# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chinanews RoBERTa model loader implementation for sequence classification (topic classification).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Chinanews RoBERTa sequence classification model variants."""

    CHINANEWS = "chinanews-chinese"


class ModelLoader(ForgeModel):
    """Chinanews RoBERTa model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.CHINANEWS: LLMModelConfig(
            pretrained_model_name="uer/roberta-base-finetuned-chinanews-chinese",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHINANEWS

    # Sample Chinese news text for topic classification
    sample_text = "北京上个月召开了两会"

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
            model="Chinanews-RoBERTa",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Chinanews RoBERTa model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Chinanews RoBERTa sequence classification."""
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
        """Decode the model output for sequence classification."""
        predicted_class_id = co_out[0].argmax().item()
        predicted_category = self.model.config.id2label[predicted_class_id]
        print(f"Predicted Category: {predicted_category}")
