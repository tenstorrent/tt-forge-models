# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RUSpam DeBERTa v4 model loader implementation for sequence classification (Russian spam detection).
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
    """Available RUSpam DeBERTa v4 model variants for sequence classification."""

    SPAM_DEBERTA_V4 = "spam_deberta_v4"


class ModelLoader(ForgeModel):
    """RUSpam DeBERTa v4 model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.SPAM_DEBERTA_V4: LLMModelConfig(
            pretrained_model_name="RUSpam/spam_deberta_v4",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPAM_DEBERTA_V4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RUSpam DeBERTa v4",
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

        sample_text = "Поздравляем! Вы выиграли новый iPhone. Перейдите по ссылке, чтобы получить приз."

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
            labels = {0: "Не спам", 1: "Спам"}
            print(
                f"Predicted label: {labels.get(predicted_class_id, predicted_class_id)}"
            )
