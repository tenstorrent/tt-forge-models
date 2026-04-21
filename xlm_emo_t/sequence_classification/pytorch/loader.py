# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XLM-EMO-T model loader implementation for sequence classification (multilingual emotion detection).
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
    """Available XLM-EMO-T model variants for sequence classification."""

    XLM_EMO_T = "MilaNLProc/xlm-emo-t"


class ModelLoader(ForgeModel):
    """XLM-EMO-T model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.XLM_EMO_T: LLMModelConfig(
            pretrained_model_name="MilaNLProc/xlm-emo-t",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLM_EMO_T

    sample_text = "I am so happy today!"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None) -> ModelInfo:
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="XLM-EMO-T",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load XLM-EMO-T model for sequence classification from Hugging Face."""

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
        """Prepare sample input for XLM-EMO-T sequence classification."""
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

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sequence classification."""
        predicted_class_id = co_out[0].argmax(-1).item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_emotion = model.config.id2label[predicted_class_id]
            print(f"Predicted Emotion: {predicted_emotion}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
