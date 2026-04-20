# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SteveTran/ob_ner_model loader implementation for token classification (NER) task.
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
    """Available SteveTran/ob_ner_model token classification variants."""

    OB_NER_MODEL = "ob_ner_model"


class ModelLoader(ForgeModel):
    """SteveTran/ob_ner_model loader for token classification (NER) task."""

    _VARIANTS = {
        ModelVariant.OB_NER_MODEL: LLMModelConfig(
            pretrained_model_name="SteveTran/ob_ner_model",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OB_NER_MODEL

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "Nike men's black leather jacket size medium in classic style."
        )
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ob_ner_model",
            variant=variant,
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
