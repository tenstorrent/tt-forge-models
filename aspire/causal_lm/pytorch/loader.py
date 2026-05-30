# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Aspire V4 ALT 8B model loader implementation for causal language modeling.

The source repository ``mradermacher/Aspire_V4_ALT-8B-Model_Stock-i1-GGUF`` only
ships GGUF (imatrix) quantizations; the upstream safetensors base model
``DreadPoor/Aspire_V4_ALT-8B-Model_Stock`` is no longer available on the Hub.
We therefore load directly from a GGUF file via the transformers ``gguf_file``
path, which reconstructs the config + tokenizer from the GGUF metadata and
dequantizes the weights. The architecture is Llama 3.1 (``general.architecture =
llama``), which is present in ``transformers.integrations.ggml.GGUF_CONFIG_MAPPING``.

We use the ``Q4_K_M`` K-quant: a standard, widely-supported GGUF quant type that
the ``gguf`` Python package can dequantize. The PCC check in the runner compares
the device output against the CPU output of this same dequantized model, so the
choice of quant level does not affect correctness validation.
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
from ....tools.utils import pad_inputs, cast_input_to_type


class ModelVariant(StrEnum):
    """Available Aspire V4 ALT model variants for causal LM."""

    V4_ALT_8B_Q4_K_M = "v4_alt_8b_q4_k_m"


class ModelLoader(ForgeModel):
    """Aspire V4 ALT 8B model loader implementation for causal LM tasks."""

    # GGUF file (within the source repo) to load for each variant.
    _GGUF_FILES = {
        ModelVariant.V4_ALT_8B_Q4_K_M: "Aspire_V4_ALT-8B-Model_Stock.i1-Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V4_ALT_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Aspire_V4_ALT-8B-Model_Stock-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V4_ALT_8B_Q4_K_M

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
            model="Aspire_V4_ALT-8B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer (reconstructed from GGUF metadata) for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Set pad token to eos token (Llama-style models have no dedicated pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Aspire V4 ALT model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype to cast inputs to.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask to the configured length
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
