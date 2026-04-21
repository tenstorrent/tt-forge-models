# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
kwoncho/ko-sroberta-multitask-informative model loader for sequence classification.

Fine-tuned Korean SRoBERTa for binary relevance classification of corporate-
related news articles. Based on jhgan/ko-sroberta-multitask.
"""

from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    """Available ko-sroberta-multitask-informative variants for sequence classification."""

    KO_SROBERTA_MULTITASK_INFORMATIVE = "kwoncho/ko-sroberta-multitask-informative"


class ModelLoader(ForgeModel):
    """kwoncho/ko-sroberta-multitask-informative model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.KO_SROBERTA_MULTITASK_INFORMATIVE: LLMModelConfig(
            pretrained_model_name="kwoncho/ko-sroberta-multitask-informative",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KO_SROBERTA_MULTITASK_INFORMATIVE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = "삼성전자가 올해 1분기 영업이익이 전년 대비 50% 증가했다고 발표했다."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ko-sroberta-multitask-informative",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ko-sroberta-multitask-informative model for sequence classification."""
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
        """Prepare a sample Korean news input for sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()
        if self.model is not None and hasattr(self.model, "config"):
            label = self.model.config.id2label.get(predicted_value, predicted_value)
            print(f"Predicted Relevance: {label}")
        else:
            print(f"Predicted class ID: {predicted_value}")
