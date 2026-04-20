# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
impresso-ad-classification-xlm-one-class loader implementation for sequence classification.

XLM-RoBERTa based multilingual genre classifier used to detect advertisement
(AD / NOT_AD) text in historical newspaper content.
"""
from typing import Optional

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
    """Available impresso-ad-classification-xlm-one-class variants for sequence classification."""

    IMPRESSO_AD_CLASSIFICATION_XLM_ONE_CLASS = (
        "impresso-ad-classification-xlm-one-class"
    )


class ModelLoader(ForgeModel):
    """impresso-ad-classification-xlm-one-class loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.IMPRESSO_AD_CLASSIFICATION_XLM_ONE_CLASS: LLMModelConfig(
            pretrained_model_name="impresso-project/impresso-ad-classification-xlm-one-class",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMPRESSO_AD_CLASSIFICATION_XLM_ONE_CLASS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="impresso-ad-classification-xlm-one-class",
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

        sample_text = (
            "Appartement 3 pièces à louer, Fr. 2'100.-/mois, tél. 079 123 45 67"
        )

        inputs = self.tokenizer(
            sample_text,
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
            print(f"Predicted class id: {predicted_class_id}")
