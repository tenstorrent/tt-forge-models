# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Indonesian RoBERTa Base Emotion Classifier (StevenLimcorn/indonesian-roberta-base-emotion-classifier) model loader implementation for sequence classification.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Indonesian RoBERTa Base Emotion Classifier model variants for sequence classification."""

    INDONESIAN_ROBERTA_BASE_EMOTION_CLASSIFIER = (
        "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
    )


class ModelLoader(ForgeModel):
    """Indonesian RoBERTa Base Emotion Classifier model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.INDONESIAN_ROBERTA_BASE_EMOTION_CLASSIFIER: LLMModelConfig(
            pretrained_model_name="StevenLimcorn/indonesian-roberta-base-emotion-classifier",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDONESIAN_ROBERTA_BASE_EMOTION_CLASSIFIER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "Hal-hal baik akan datang."
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Indonesian_RoBERTa_Base_Emotion_Classifier",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Indonesian RoBERTa Base Emotion Classifier model for sequence classification from Hugging Face."""
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
        """Prepare sample input for Indonesian RoBERTa Base Emotion Classifier sequence classification."""
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
        """Decode the model output for emotion classification."""
        predicted_class_id = co_out[0].argmax(-1).item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_emotion = model.config.id2label[predicted_class_id]
            print(f"Predicted Emotion: {predicted_emotion}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
