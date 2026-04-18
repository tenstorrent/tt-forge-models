# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unslothai 2 model loader implementation for feature extraction.

Unslothai 2 is a Llama-based feature extraction model from the Unsloth AI team.
"""
import torch
from transformers import AutoModel, AutoTokenizer, LlamaConfig
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
    """Available Unslothai 2 model variants for feature extraction."""

    UNSLOTHAI_2 = "unslothai/2"


class ModelLoader(ForgeModel):
    """Unslothai 2 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.UNSLOTHAI_2: LLMModelConfig(
            pretrained_model_name="unslothai/2",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTHAI_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Unslothai 2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # The unslothai/2 repo has no tokenizer; use a compatible Llama tokenizer.
    _TOKENIZER_SOURCE = "huggyllama/llama-7b"

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self._TOKENIZER_SOURCE)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Unslothai 2 model from Hugging Face."""
        if self.tokenizer is None:
            self._load_tokenizer()

        # The upstream config has all-zero dimensions; override with small values.
        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_hidden_layers=2,
            num_key_value_heads=2,
            vocab_size=32000,
        )
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = AutoModel.from_config(config)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, query=None):
        """Load and return sample inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if query is None:
            query = "What is the capital of France?"

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode the model output for feature extraction."""
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor."""
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
