# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Q7-NeonPixel model loader implementation for sequence classification.
"""
from typing import Optional

from transformers import BertForSequenceClassification, BertTokenizer

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Q7-NeonPixel model variants for sequence classification."""

    Q7_NEONPIXEL_MODEL = "Q7_NeonPixel_Model"


class ModelLoader(ForgeModel):
    """Q7-NeonPixel model loader implementation for sequence classification tasks."""

    _VARIANTS = {
        ModelVariant.Q7_NEONPIXEL_MODEL: LLMModelConfig(
            pretrained_model_name="TextAsData/Q7-NeonPixel-model",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q7_NEONPIXEL_MODEL

    _SAMPLE_TEXTS = {
        ModelVariant.Q7_NEONPIXEL_MODEL: "Please review the attached quarterly sales report and provide feedback.",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.review = self._SAMPLE_TEXTS.get(
            self._variant,
            "Please review the attached quarterly sales report and provide feedback.",
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Q7-NeonPixel",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Q7-NeonPixel model for sequence classification from Hugging Face."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Q7-NeonPixel sequence classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sequence classification."""
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Class: {self.model.config.id2label[predicted_value]}")
