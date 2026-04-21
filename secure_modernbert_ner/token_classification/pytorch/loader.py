# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SecureModernBERT-NER model loader implementation for token classification.
ModernBERT-large fine-tuned for cybersecurity named entity recognition over
22 threat-intelligence classes (threat actors, malware, CVEs, IOCs, etc.).
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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
    """Available SecureModernBERT-NER model variants for token classification."""

    SECURE_MODERNBERT_NER = "attack-vector/SecureModernBERT-NER"


class ModelLoader(ForgeModel):
    """SecureModernBERT-NER model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.SECURE_MODERNBERT_NER: LLMModelConfig(
            pretrained_model_name="attack-vector/SecureModernBERT-NER",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SECURE_MODERNBERT_NER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = (
            "TrickBot connects to hxxp://185.222.202.55 to exfiltrate data "
            "from Windows hosts."
        )
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SecureModernBERT-NER",
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
