# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RuBERT Tiny2 Russian Emotion Detection model loader implementation for emotion detection.
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
    """Available RuBERT Tiny2 Russian Emotion Detection model variants."""

    DJACON_RUBERT_TINY2_RUSSIAN_EMOTION_DETECTION = (
        "djacon_rubert-tiny2-russian-emotion-detection"
    )


class ModelLoader(ForgeModel):
    """RuBERT Tiny2 Russian Emotion Detection model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DJACON_RUBERT_TINY2_RUSSIAN_EMOTION_DETECTION: LLMModelConfig(
            pretrained_model_name="Djacon/rubert-tiny2-russian-emotion-detection",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DJACON_RUBERT_TINY2_RUSSIAN_EMOTION_DETECTION

    # Sample Russian text for emotion detection
    sample_text = "Я очень рад видеть вас!"

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="RuBERT-Tiny2-Russian-Emotion-Detection",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RuBERT Tiny2 Russian Emotion Detection model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RuBERT Tiny2 Russian Emotion Detection model instance.
        """
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
        """Prepare sample input for RuBERT Tiny2 Russian Emotion Detection.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
        """Decode the model output for emotion detection.

        Args:
            co_out: Model output
            framework_model: Optional model to use for label lookup
        """
        predicted_class_id = co_out[0].argmax(-1).item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_emotion = model.config.id2label[predicted_class_id]
            print(f"Predicted Emotion: {predicted_emotion}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
