# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multilingual PII NER (Ar86Bat/multilang-pii-ner) model loader implementation
for token classification.

XLM-RoBERTa-base fine-tuned for Named Entity Recognition to detect and mask
Personally Identifiable Information across English, German, French, and
Italian text.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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
    """Available Multilingual PII NER model variants for token classification."""

    AR86BAT_MULTILANG_PII_NER = "Ar86Bat/multilang-pii-ner"


class ModelLoader(ForgeModel):
    """Multilingual PII NER model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.AR86BAT_MULTILANG_PII_NER: LLMModelConfig(
            pretrained_model_name="Ar86Bat/multilang-pii-ner",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AR86BAT_MULTILANG_PII_NER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "John Doe was born on 12/12/1990 and lives in Berlin."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Multilingual PII NER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
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
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"Predicted Labels: {predicted_tokens_classes}")
