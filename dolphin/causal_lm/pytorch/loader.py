# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dolphin (Llama 3.1 8B) GGUF model loader implementation for causal language modeling.

The dphn/dolphin-2.9.4-llama3.1-8b-gguf repository ships only quantized GGUF
weights. Transformers dequantizes the selected GGUF file back into full-precision
torch tensors via the ``gguf_file`` argument, yielding a standard
``LlamaForCausalLM`` module. The tokenizer is also embedded in the GGUF file.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Dolphin GGUF model variants for causal LM."""

    DOLPHIN_2_9_4_LLAMA31_8B_Q4_K_M = "2.9.4_llama3.1_8b_q4_k_m"


class ModelLoader(ForgeModel):
    """Dolphin GGUF model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOLPHIN_2_9_4_LLAMA31_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="dphn/dolphin-2.9.4-llama3.1-8b-gguf",
            max_length=128,
        ),
    }

    # GGUF filename within the repository for each variant.
    _GGUF_FILES = {
        ModelVariant.DOLPHIN_2_9_4_LLAMA31_8B_Q4_K_M: (
            "dolphin-2.9.4-llama3.1-8b-Q4_K_M.gguf"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOLPHIN_2_9_4_LLAMA31_8B_Q4_K_M

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
            model="Dolphin",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from the embedded GGUF tokenizer.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Set pad token to eos token for Llama-based models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dolphin model instance for this instance's variant.

        The GGUF weights are dequantized into a standard ``LlamaForCausalLM``.

        Args:
            dtype_override: Optional torch.dtype to cast the model weights to.
                If not provided, the dequantized GGUF dtype (float32) is used.

        Returns:
            torch.nn.Module: The Dolphin model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dolphin model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype to cast float inputs to.
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

        # Pad input_ids and attention_mask to a fixed sequence length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the Dolphin model variant.

        Returns:
            The configuration object for the Dolphin model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )

        return self.config
