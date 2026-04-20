# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT Base Uncased City Country NER model loader implementation for token classification.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    """Available BERT Base Uncased City Country NER model variants."""

    BERT_BASE_UNCASED_CITY_COUNTRY_NER = "ml6team/bert-base-uncased-city-country-ner"


class ModelLoader(ForgeModel):
    """BERT Base Uncased City Country NER model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.BERT_BASE_UNCASED_CITY_COUNTRY_NER: LLMModelConfig(
            pretrained_model_name="ml6team/bert-base-uncased-city-country-ner",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_BASE_UNCASED_CITY_COUNTRY_NER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "My name is Kermit and I live in London."
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="BERT Base Uncased City Country NER",
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
        print(f"NER Tags: {predicted_tokens_classes}")
