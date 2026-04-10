# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bert Base Personality model loader implementation for sequence classification."""
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Minej/bert-base-personality",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    sample_text_a = "How similar are these two passages?"
    sample_text_b = "They are semantically related."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Bert Base Personality",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._variant_config.pretrained_model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModelForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        tokenizer = self._load_tokenizer()
        return tokenizer(
            self.sample_text_a,
            self.sample_text_b,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )

    def decode_output(self, outputs, **kwargs):
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs
