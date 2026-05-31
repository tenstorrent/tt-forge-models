# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DnD GGUF Causal LM model loader implementation.

The HuggingFace repo ``elusinchi/dnd-gguf-models`` ships only quantized GGUF
weights (llama.cpp format) for a Dungeons & Dragons assistant. It contains
several Qwen-architecture base models plus many LoRA adapters. There are no
safetensors / config.json files at the repo root, so the models are loaded via
transformers' GGUF support: ``from_pretrained(..., gguf_file=<path>)`` parses
the GGUF metadata to reconstruct the HF config/tokenizer and de-quantizes the
weights to float32 on load.
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
    """Available DnD GGUF base-model variants for causal language modeling."""

    QWEN_2_5_0_5B = "qwen2_5_0_5b"
    QWEN_3_5_0_8B = "qwen3_5_0_8b"
    QWEN_3_5_4B = "qwen3_5_4b"


# Single HuggingFace repository hosting every GGUF artifact.
_REPO_ID = "elusinchi/dnd-gguf-models"

# Relative path (within the repo) of the GGUF base-weights file for each variant.
_GGUF_FILES = {
    ModelVariant.QWEN_2_5_0_5B: "qwen2.5-0.5b/base_q4_k_m.gguf",
    ModelVariant.QWEN_3_5_0_8B: "qwen3.5-0.8b/base_q4_k_m.gguf",
    ModelVariant.QWEN_3_5_4B: "qwen3.5-4b/base_q4_k_m.gguf",
}


class ModelLoader(ForgeModel):
    """DnD GGUF model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs.
    # ``pretrained_model_name`` is the HF repo id; the specific GGUF file is
    # selected via ``_GGUF_FILES`` in load_model/_load_tokenizer.
    _VARIANTS = {
        ModelVariant.QWEN_2_5_0_5B: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_0_8B: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B: LLMModelConfig(
            pretrained_model_name=_REPO_ID,
            max_length=128,
        ),
    }

    # Default variant to use: the smallest, fully validated base model.
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_0_5B

    # Shared configuration parameters (D&D-themed prompt for a dungeon master).
    sample_text = "You are a dungeon master. Describe the dark forest the party enters."

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
            model="DnD GGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        """Return the GGUF file path within the repo for this instance's variant."""
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        """Load the tokenizer (reconstructed from the GGUF metadata).

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DnD GGUF model instance for this instance's variant.

        The GGUF weights are de-quantized to float32 during loading. When
        ``dtype_override`` is provided the model is cast to that dtype afterwards.

        Args:
            dtype_override: Optional torch.dtype to cast the model to. If not
                provided, the de-quantized float32 weights are used.

        Returns:
            torch.nn.Module: The model instance for causal language modeling.
        """
        # Ensure tokenizer is loaded.
        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
            **kwargs,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DnD GGUF model.

        Args:
            dtype_override: Unused for tokenized (integer) inputs; accepted for
                interface compatibility.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized.
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension by repeating the single prompt.
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object reconstructed from the GGUF metadata.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file(),
        )
        return self.config
