# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
RoBERTa model loader implementation for masked language modeling.
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
    """Available RoBERTa model variants."""

    BASE = "base"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """RoBERTa model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="FacebookAI/roberta-base",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="FacebookAI/roberta-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
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
            model="roberta",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the tokenizer for the model.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            Tokenizer: The tokenizer for the model
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the RoBERTa model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import FlaxRobertaForMaskedLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load model
        model = FlaxRobertaForMaskedLM.from_pretrained(
            self._model_name,
            dtype=dtype_override,
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return the inputs for the model.

        Args:
            dtype_override: Optional dtype to override the default dtype.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
