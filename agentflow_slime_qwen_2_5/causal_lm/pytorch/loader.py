# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AgentFlow Slime Agentic Qwen2.5 7B (GGUF) causal LM loader implementation.

The upstream weights are distributed only as GGUF quantizations
(mradermacher/AgentFlow_Slime_Agentic_Qwen2.5_7B-i1-GGUF). transformers
dequantizes the selected GGUF file on load and reconstructs a standard
Qwen2 causal LM, so the rest of the bringup flow is identical to other
Qwen2.5 causal LM loaders.
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
    """Available AgentFlow Slime Qwen2.5 GGUF variants."""

    QWEN_2_5_7B_Q4_K_M = "7B_Q4_K_M"


class ModelLoader(ForgeModel):
    """AgentFlow Slime Agentic Qwen2.5 7B GGUF loader for causal language modeling."""

    # GGUF file (inside the HF repo) to dequantize for each variant.
    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_7B_Q4_K_M: "AgentFlow_Slime_Agentic_Qwen2.5_7B.i1-Q4_K_M.gguf",
    }

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_5_7B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/AgentFlow_Slime_Agentic_Qwen2.5_7B-i1-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_7B_Q4_K_M

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language models."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

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
            model="AgentFlow Slime Qwen2.5 7B",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self):
        """Return the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the AgentFlow Slime Qwen2.5 model instance.

        The GGUF weights are dequantized by transformers into a standard
        Qwen2ForCausalLM module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Qwen2 causal LM model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Unused for integer token inputs; kept for interface parity.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.config
