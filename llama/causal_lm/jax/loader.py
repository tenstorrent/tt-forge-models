# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Llama model loader implementation for causal language modeling.
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
    """Available Llama model variants."""

    _3B_V2 = "3b-v2"


class ModelLoader(ForgeModel):
    """Llama model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant._3B_V2: LLMModelConfig(
            pretrained_model_name="openlm-research/open_llama_3b_v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant._3B_V2

    sample_text = "Hello there fellow traveller"

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
            model="llama",
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

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Llama model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxLlamaForCausalLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxLlamaForCausalLM.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Llama model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Create tokenized inputs
        inputs = self._tokenizer(self.sample_text, return_tensors="jax")

        return inputs
