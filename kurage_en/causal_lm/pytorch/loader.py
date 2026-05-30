# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kurage En model loader implementation for causal language modeling.

Kurage En (lightblue/kurage-en) is a Qwen2-7B based RAG model. This loader
consumes the GGUF distribution published at bartowski/kurage-en-GGUF, which
reconstructs the config, tokenizer and (dequantized) weights directly from the
GGUF metadata via transformers' GGUF support.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available Kurage En model variants for causal language modeling."""

    KURAGE_EN = "kurage_en"


class ModelLoader(ForgeModel):
    """Kurage En model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.KURAGE_EN: LLMModelConfig(
            pretrained_model_name="bartowski/kurage-en-GGUF",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.KURAGE_EN

    # GGUF file to load from the GGUF repo. A K-quant is used (the dequantized
    # golden weights only change the CPU reference, not the device-vs-CPU PCC).
    GGUF_FILE = "kurage-en-Q4_K_M.gguf"

    # Shared configuration parameters
    sample_text = (
        "Using the provided documents, answer the question as accurately as "
        "possible. Question: What are the main benefits of retrieval augmented "
        "generation for large language models, and how does it reduce "
        "hallucinations in practice?"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

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
            model="Kurage En",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kurage En model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Kurage En model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # The GGUF dequantization path does not honor torch_dtype for every
        # tensor, so make dtype_override authoritative.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Kurage En model with this instance's variant settings.

        Args:
            dtype_override: Unused for token inputs (kept for interface compatibility).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Use the Qwen2 chat template
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
        """Load and return the configuration for the Kurage En model variant.

        Returns:
            The configuration object for the Kurage En model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )

        return self.config
