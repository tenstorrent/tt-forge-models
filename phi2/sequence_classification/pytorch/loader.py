# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI2 model loader implementation for sequence classification.
"""
import torch
from transformers import PhiForSequenceClassification, PhiConfig, AutoTokenizer
from typing import Optional

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
    """Available PHI2 model variants."""

    PHI2 = "microsoft/phi-2"
    PHI2_PYTDML = "microsoft/phi-2-pytdml"


class ModelLoader(ForgeModel):
    """PHI2 model loader implementation for sequence classification tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.PHI2: LLMModelConfig(
            pretrained_model_name="microsoft/phi-2",
            max_length=128,
        ),
        ModelVariant.PHI2_PYTDML: LLMModelConfig(
            pretrained_model_name="microsoft/phi-2-pytdml",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI2

    # Shared configuration parameters
    sample_text = "I am not satisfied with the quality of this product."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        # Determine group based on variant
        if variant == ModelVariant.PHI2:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="phi2",
            variant=variant,
            group=group,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the PHI2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The PHI2 model instance for sequence classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load PHI2 config with custom settings for sequence classification
        config = PhiConfig.from_pretrained(pretrained_model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = False
        config_dict["pad_token_id"] = None
        config = PhiConfig(**config_dict)

        # Load the model with dtype override if specified
        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = PhiForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the PHI2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the sequence classification task
        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Predicted sentiment label
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        # Get the logits from the outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get the predicted class ID
        predicted_value = logits.argmax(-1).item()

        # Get the model to access config
        model = self.load_model()
        predicted_sentiment = model.config.id2label[predicted_value]

        return predicted_sentiment
