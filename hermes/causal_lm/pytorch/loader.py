# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hermes-3 (Llama-3.1) GGUF model loader implementation for causal language modeling.

This loader targets the bartowski GGUF distribution of NousResearch's
Hermes-3-Llama-3.1-8B model. The model is a standard Llama-3.1 architecture; the
GGUF file is dequantized on load by HuggingFace transformers (via the ``gguf_file``
argument) into ordinary float weights, so it runs through the regular
``AutoModelForCausalLM`` path.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Hermes-3 GGUF model variants for causal LM."""

    # Quantized GGUF distribution from bartowski. The Q4_K_M quant is the
    # commonly used balanced quant and is dequantized by transformers on load.
    HERMES_3_LLAMA_3_1_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """Hermes-3 (Llama-3.1) GGUF model loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HERMES_3_LLAMA_3_1_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/Hermes-3-Llama-3.1-8B-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HERMES_3_LLAMA_3_1_8B_Q4_K_M

    # Mapping of variant -> GGUF filename within the repo. transformers uses this
    # to select and dequantize the correct quantized checkpoint.
    _GGUF_FILES = {
        ModelVariant.HERMES_3_LLAMA_3_1_8B_Q4_K_M: "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
    }

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Hermes-3-Llama-3.1-8B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF filename to load for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the GGUF checkpoint.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token for Llama-style models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Hermes-3 model instance for this instance's variant.

        The GGUF file is dequantized by transformers into standard float weights.

        Args:
            dtype_override: Optional torch.dtype to cast the model weights to.
                            If not provided, the dequantized default dtype is used.

        Returns:
            torch.nn.Module: The Hermes-3 model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # GGUF weights are dequantized to a default float dtype; cast if requested.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Hermes-3 model.

        Args:
            dtype_override: Optional torch.dtype applied to the input tensors.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Keep the prefill sequence at the natural prompt length. Padding the
        # input with a large block of masked/zero positions only adds logits over
        # meaningless tokens, which inflates the bf16 numerical error in the
        # device-vs-CPU comparison without adding test coverage.
        self.seq_len = inputs["input_ids"].shape[-1]
        return inputs
