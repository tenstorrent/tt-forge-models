# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FrugalScore Tiny BERT model loader implementation for sequence classification.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    """Available FrugalScore Tiny BERT model variants for sequence classification."""

    FRUGALSCORE_TINY_BERT_BASE_BERT_SCORE = "frugalscore_tiny_bert-base_bert-score"


class ModelLoader(ForgeModel):
    """FrugalScore Tiny BERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.FRUGALSCORE_TINY_BERT_BASE_BERT_SCORE: LLMModelConfig(
            pretrained_model_name="moussaKam/frugalscore_tiny_bert-base_bert-score",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FRUGALSCORE_TINY_BERT_BASE_BERT_SCORE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.reference_text = "The cat is sitting on the mat."
        self.candidate_text = "A cat sits on the mat."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="FrugalScore-Tiny-BERT",
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
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.reference_text,
            self.candidate_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_score = co_out[0].squeeze().item()
        print(f"Predicted FrugalScore: {predicted_score:.4f}")
