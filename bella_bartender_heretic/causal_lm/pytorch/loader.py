# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bella Bartender Heretic 3B model loader implementation for causal language modeling.

This loader brings up the GGUF-quantized release
``mradermacher/bella-bartender-heretic-3b-i1-GGUF`` (a Llama-3.2-3B fine-tune).
The GGUF weights are loaded through transformers' native GGUF support
(``gguf_file=...``), which de-quantizes them into a standard
``LlamaForCausalLM`` PyTorch model that the TT compiler can consume.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
from ....tools.utils import cast_input_to_type, pad_inputs


class ModelVariant(StrEnum):
    """Available Bella Bartender Heretic variants for causal LM."""

    # imatrix Q4_K_M GGUF quantization of the 3B fine-tune.
    BELLA_3B_I1_Q4_K_M = "3b_i1_Q4_K_M"


class ModelLoader(ForgeModel):
    """Bella Bartender Heretic model loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BELLA_3B_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/bella-bartender-heretic-3b-i1-GGUF",
            max_length=128,
        ),
    }

    # GGUF file within the repo to de-quantize for each variant.
    _GGUF_FILES = {
        ModelVariant.BELLA_3B_I1_Q4_K_M: "bella-bartender-heretic-3b.i1-Q4_K_M.gguf",
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BELLA_3B_I1_Q4_K_M

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
        return ModelInfo(
            model="BellaBartenderHeretic",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename to de-quantize for this instance's variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Bella Bartender Heretic model instance.

        The GGUF weights are de-quantized into a standard ``LlamaForCausalLM``.

        Args:
            dtype_override: Optional torch.dtype to override the de-quantized
                            dtype. If not provided, weights are de-quantized to
                            float32.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"torch_dtype": torch.float32}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file(), **model_kwargs
        )

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

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

        # Pad input_ids and attention_mask to a fixed length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the Llama model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file()
        )

        return self.config
