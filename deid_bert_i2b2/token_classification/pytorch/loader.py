# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OBI DEID BERT i2b2 (obi/deid_bert_i2b2) model loader implementation for
token classification.

ClinicalBERT fine-tuned on the i2b2 2014 corpus to tag Protected Health
Information (PHI) spans in medical notes across 11 categories (DATE, STAFF,
HOSP, AGE, LOC, PATIENT, PHONE, ID, PATORG, EMAIL, OTHERPHI).
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
    """Available OBI DEID BERT i2b2 model variants for token classification."""

    OBI_DEID_BERT_I2B2 = "obi/deid_bert_i2b2"


class ModelLoader(ForgeModel):
    """OBI DEID BERT i2b2 model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.OBI_DEID_BERT_I2B2: LLMModelConfig(
            pretrained_model_name="obi/deid_bert_i2b2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OBI_DEID_BERT_I2B2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = (
            "Dr. John Smith treated patient Jane Doe at Stanford Medical "
            "Center on 03/15/2024."
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="OBI DEID BERT i2b2",
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
