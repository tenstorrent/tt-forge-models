# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-R1-Distill-Llama-8B (GGUF) causal LM model loader implementation.

Loads the unsloth GGUF distribution of DeepSeek-R1-Distill-Llama-8B. transformers
reads the quantized GGUF file via the ``gguf_file`` argument and dequantizes the
weights back to the base ``LlamaForCausalLM`` architecture at load time.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    """Available DeepSeek-R1-Distill-Llama-8B GGUF variants for causal LM."""

    DEEPSEEK_R1_DISTILL_LLAMA_8B_Q4_K_M = "8B-Q4_K_M"


class ModelLoader(ForgeModel):
    """DeepSeek-R1-Distill-Llama-8B GGUF loader for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_R1_DISTILL_LLAMA_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
            max_length=128,
        ),
    }

    # GGUF file (within the repo) to load for each variant. transformers
    # dequantizes this back to the base LlamaForCausalLM architecture.
    _GGUF_FILES = {
        ModelVariant.DEEPSEEK_R1_DISTILL_LLAMA_8B_Q4_K_M: (
            "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_R1_DISTILL_LLAMA_8B_Q4_K_M

    # Sample text for causal LM
    sample_text = "Who are you?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

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
            model="DeepSeek-R1-Distill-Llama-8B (GGUF)",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        tokenizer_kwargs = {"gguf_file": gguf_file}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepSeek-R1-Distill-Llama-8B GGUF model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the dequantized model uses its default dtype.

        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
        """Load and return sample inputs for the model with this instance's variant settings.

        The natural (unpadded) prompt is returned with an all-ones attention mask;
        padding the prompt out to a fixed sequence length introduces ill-defined
        logits at the padded positions that depress device-vs-host PCC.

        Args:
            dtype_override: Optional torch.dtype (unused for integer token inputs).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
