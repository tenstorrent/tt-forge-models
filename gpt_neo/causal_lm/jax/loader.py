# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-Neo model loader implementation for causal language modeling.
"""

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
    """Available GPT-Neo model variants."""

    GPT_NEO_125M = "_125m"
    GPT_NEO_1_3B = "_1_3b"
    GPT_NEO_2_7B = "_2_7b"


class ModelLoader(ForgeModel):
    """GPT-Neo model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GPT_NEO_125M: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-125M",
        ),
        ModelVariant.GPT_NEO_1_3B: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-1.3B",
        ),
        ModelVariant.GPT_NEO_2_7B: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-2.7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_NEO_1_3B

    sample_text = "Hello there fellow traveler"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="gpt-neo",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.
        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.
        Returns:
            tokenizer: The loaded tokenizer instance
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the GPT-Neo model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxGPTNeoForCausalLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxGPTNeoForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the GPT-Neo model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
