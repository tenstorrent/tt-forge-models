# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-SEA-LION model loader implementation for causal language modeling.

SEA-LION v3.5 8B-R is a Llama-3.1 architecture model fine-tuned by AI Singapore
for Southeast-Asian languages. The weights are distributed exclusively in GGUF
format, so this loader dequantizes a GGUF checkpoint into a standard PyTorch
``LlamaForCausalLM`` via the transformers GGUF integration. The chosen
quantization level only affects fidelity, not graph structure: the CPU golden
and the device run use the same dequantized weights, so PCC is unaffected.
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
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available Llama-SEA-LION model variants for causal LM."""

    LLAMA_SEA_LION_V3_5_8B_R = "v3.5_8B_R"


class ModelLoader(ForgeModel):
    """Llama-SEA-LION model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LLAMA_SEA_LION_V3_5_8B_R: LLMModelConfig(
            pretrained_model_name="aisingapore/Llama-SEA-LION-v3.5-8B-R-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_SEA_LION_V3_5_8B_R

    # GGUF checkpoint to dequantize for each variant. The Q4_K_M file keeps the
    # download small (~4.9GB); transformers dequantizes it to fp32 PyTorch
    # tensors on load.
    _GGUF_FILES = {
        ModelVariant.LLAMA_SEA_LION_V3_5_8B_R: "Llama-SEA-LION-v3.5-8B-R-Q4_K_M.gguf",
    }

    # Sample text for causal LM
    sample_text = "What are the main islands of Southeast Asia?"

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
            model="Llama-SEA-LION",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF checkpoint.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )

        # Set pad token to eos token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Llama-SEA-LION model instance for this variant.

        The GGUF checkpoint is dequantized into a standard ``LlamaForCausalLM``.

        Args:
            dtype_override: Optional torch.dtype to cast the model weights to.
                            If not provided, the dequantized fp32 weights are used.

        Returns:
            torch.nn.Module: The Llama-SEA-LION model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
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
        """Load and return sample inputs for the Llama-SEA-LION model.

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
