# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Loader for GGUF causal-LM checkpoints published in the
``chanwoocho/asplos2026`` HuggingFace repository.

The repository ships llama.cpp GGUF quantizations only (no safetensors).
transformers' GGUF support dequantizes these back into the standard PyTorch
architecture (Llama, Qwen2, etc.), selected per variant via the ``gguf_file``
argument, so a single causal-LM loader covers every checkpoint in the repo.
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

# Shared HuggingFace repository for every GGUF checkpoint.
_REPO = "chanwoocho/asplos2026"


class ModelVariant(StrEnum):
    """Available GGUF causal-LM variants from chanwoocho/asplos2026."""

    LLAMA_3_2_1B_BF16 = "llama_3_2_1b_bf16"
    QWEN2_5_0_5B_BF16 = "qwen2_5_0_5b_bf16"
    TINYLLAMA_1_1B_BF16 = "tinyllama_1_1b_bf16"


# Map each variant to the GGUF file it loads from the shared repository.
_GGUF_FILES = {
    ModelVariant.LLAMA_3_2_1B_BF16: "Llama-3.2-1B-BF16.gguf",
    ModelVariant.QWEN2_5_0_5B_BF16: "Qwen2.5-0.5B-BF16.gguf",
    ModelVariant.TINYLLAMA_1_1B_BF16: "TinyLlama-1.1B-BF16.gguf",
}


class ModelLoader(ForgeModel):
    """Loader for GGUF causal-LM checkpoints from chanwoocho/asplos2026."""

    # All variants share the same repo; the GGUF file is selected in load_model.
    _VARIANTS = {
        ModelVariant.LLAMA_3_2_1B_BF16: LLMModelConfig(
            pretrained_model_name=_REPO,
            max_length=128,
        ),
        ModelVariant.QWEN2_5_0_5B_BF16: LLMModelConfig(
            pretrained_model_name=_REPO,
            max_length=128,
        ),
        ModelVariant.TINYLLAMA_1_1B_BF16: LLMModelConfig(
            pretrained_model_name=_REPO,
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B_BF16

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
            model="asplos2026",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF filename for this instance's variant."""
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant from its GGUF file.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for these decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the causal-LM model instance for this variant.

        The GGUF checkpoint is dequantized by transformers into the standard
        PyTorch architecture indicated by the file's embedded config.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The causal-LM model instance.
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
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this variant's settings.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
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
        """Load and return the configuration for the current variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )

        return self.config
