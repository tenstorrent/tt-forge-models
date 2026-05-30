# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepHermes-3 (Llama-3 8B) GGUF model loader implementation for causal language modeling.

The source repository (bartowski/NousResearch_DeepHermes-3-Llama-3-8B-Preview-GGUF)
ships only quantized GGUF files; there is no config.json or tokenizer on the repo.
transformers reconstructs the Llama config and tokenizer from the GGUF metadata and
dequantizes the weights to torch tensors when ``gguf_file`` is passed to
``from_pretrained``.
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
    """Available DeepHermes-3 GGUF model variants for causal LM."""

    # 8B Llama-3 finetune, Q4_K_M quantization (general-purpose recommended quant).
    DEEPHERMES_3_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """DeepHermes-3 GGUF model loader implementation for causal language modeling tasks."""

    # GGUF filename to load for each variant (relative to the repo root).
    _GGUF_FILES = {
        ModelVariant.DEEPHERMES_3_8B_Q4_K_M: (
            "NousResearch_DeepHermes-3-Llama-3-8B-Preview-Q4_K_M.gguf"
        ),
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPHERMES_3_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/NousResearch_DeepHermes-3-Llama-3-8B-Preview-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPHERMES_3_8B_Q4_K_M

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
            model="DeepHermes3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant from the GGUF file.

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

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the DeepHermes-3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the dequantized weight
                            dtype. If not provided, weights are dequantized to float32.

        Returns:
            torch.nn.Module: The DeepHermes-3 model instance for causal LM.
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
        """Load and return sample inputs for the DeepHermes-3 model.

        Args:
            dtype_override: Optional torch.dtype to cast the inputs to.
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
