# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BioClinical-ModernBERT model loader implementation for sequence classification.
"""
from typing import Optional

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BioClinical-ModernBERT model variants for sequence classification."""

    SYMPTOM2DISEASE_DROPOUT_42 = "Symptom2Disease_Dropout_42"


class ModelLoader(ForgeModel):
    """BioClinical-ModernBERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SYMPTOM2DISEASE_DROPOUT_42: LLMModelConfig(
            pretrained_model_name="notlath/BioClinical-ModernBERT-base-Symptom2Disease_WITH-DROPOUT-42",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SYMPTOM2DISEASE_DROPOUT_42

    _SAMPLE_TEXTS = {
        ModelVariant.SYMPTOM2DISEASE_DROPOUT_42: (
            "I have a persistent cough, fever, and shortness of breath for the past three days.",
        ),
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BioClinical-ModernBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        sample = self._SAMPLE_TEXTS.get(
            self._variant,
            (
                "I have a persistent cough, fever, and shortness of breath for the past three days.",
            ),
        )

        inputs = self.tokenizer(
            *sample,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_label = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
