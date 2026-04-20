# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERTimbau Brazilian Court Decisions model loader implementation for sequence classification.
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
    """Available BERTimbau Brazilian Court Decisions model variants for sequence classification."""

    BERTIMBAU_BASE_FINETUNED_BRAZILIAN_COURT_DECISIONS = (
        "Luciano/bertimbau-base-finetuned-brazilian_court_decisions"
    )


class ModelLoader(ForgeModel):
    """BERTimbau Brazilian Court Decisions model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BERTIMBAU_BASE_FINETUNED_BRAZILIAN_COURT_DECISIONS: LLMModelConfig(
            pretrained_model_name="Luciano/bertimbau-base-finetuned-brazilian_court_decisions",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERTIMBAU_BASE_FINETUNED_BRAZILIAN_COURT_DECISIONS

    sample_text = (
        "Trata-se de recurso interposto pela parte autora em face de decisão "
        "que julgou improcedente o pedido de indenização por danos morais."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="BERTimbau_Brazilian_Court_Decisions",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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
        predicted_class_id = co_out[0].argmax(-1).item()
        label = self.model.config.id2label[predicted_class_id]
        print(f"Predicted Label: {label}")
        return label
