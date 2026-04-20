# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Clinical Assertion Negation BERT model loader implementation for sequence classification.
"""

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
    """Available Clinical Assertion Negation BERT model variants for sequence classification."""

    CLINICAL_ASSERTION_NEGATION_BERT = "clinical_assertion_negation_bert"


class ModelLoader(ForgeModel):
    """Clinical Assertion Negation BERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.CLINICAL_ASSERTION_NEGATION_BERT: LLMModelConfig(
            pretrained_model_name="bvanaken/clinical-assertion-negation-bert",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CLINICAL_ASSERTION_NEGATION_BERT

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = (
            "The patient recovered during the night and now denies any "
            "[entity] shortness of breath [entity]."
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Clinical_Assertion_Negation_BERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Clinical Assertion Negation BERT model for sequence classification from Hugging Face."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Clinical Assertion Negation BERT sequence classification."""
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
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()

        print(f"Predicted Assertion: {self.model.config.id2label[predicted_value]}")
