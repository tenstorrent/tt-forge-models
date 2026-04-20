# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mDeBERTa V3 model loader implementation for zero-shot classification.

Uses Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7, a multilingual
NLI model fine-tuned from microsoft/mdeberta-v3-base that can perform
zero-shot classification across many languages via entailment scoring.
"""
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available mDeBERTa V3 model variants for zero-shot classification."""

    XENOVA_MDEBERTA_V3_BASE_XNLI_MULTILINGUAL_NLI_2MIL7 = (
        "Xenova_mDeBERTa_V3_Base_Xnli_Multilingual_Nli_2mil7"
    )


class ModelLoader(ForgeModel):
    """mDeBERTa V3 model loader implementation for zero-shot classification."""

    _VARIANTS = {
        ModelVariant.XENOVA_MDEBERTA_V3_BASE_XNLI_MULTILINGUAL_NLI_2MIL7: ModelConfig(
            pretrained_model_name="Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XENOVA_MDEBERTA_V3_BASE_XNLI_MULTILINGUAL_NLI_2MIL7

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mDeBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        text = "Angela Merkel is a politician in Germany and target of a conspiracy theory."
        hypothesis = "This text is about politics."

        inputs = self.tokenizer(
            text,
            hypothesis,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        predicted_class_id = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_id]
        print(f"Predicted: {predicted_label}")
