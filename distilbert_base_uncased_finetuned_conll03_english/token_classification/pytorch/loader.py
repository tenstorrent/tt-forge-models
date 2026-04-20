# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
elastic/distilbert-base-uncased-finetuned-conll03-english model loader
implementation for token classification (NER) task.
"""
from typing import Optional

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
    """Available elastic/distilbert-base-uncased-finetuned-conll03-english variants."""

    DISTILBERT_BASE_UNCASED_FINETUNED_CONLL03_ENGLISH = (
        "elastic/distilbert-base-uncased-finetuned-conll03-english"
    )


class ModelLoader(ForgeModel):
    """elastic/distilbert-base-uncased-finetuned-conll03-english token classification loader."""

    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_UNCASED_FINETUNED_CONLL03_ENGLISH: LLMModelConfig(
            pretrained_model_name="elastic/distilbert-base-uncased-finetuned-conll03-english",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_UNCASED_FINETUNED_CONLL03_ENGLISH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = "HuggingFace is a company based in Paris and New York"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="distilbert-base-uncased-finetuned-conll03-english",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the elastic/distilbert-base-uncased-finetuned-conll03-english model."""
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
        """Prepare sample input for token classification."""
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
        """Decode the model output for token classification."""
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
