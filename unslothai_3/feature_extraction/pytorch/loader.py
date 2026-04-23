# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unslothai 3 model loader implementation for feature extraction.

Unslothai 3 is a Llama-based feature extraction model from the Unsloth AI team.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaConfig, LlamaModel
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

# unslothai/3 has no tokenizer files; use a compatible public Llama 1/2 tokenizer
_TOKENIZER_NAME = "huggyllama/llama-7b"


class ModelVariant(StrEnum):
    """Available Unslothai 3 model variants for feature extraction."""

    UNSLOTHAI_3 = "unslothai/3"


class ModelLoader(ForgeModel):
    """Unslothai 3 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.UNSLOTHAI_3: LLMModelConfig(
            pretrained_model_name="unslothai/3",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTHAI_3

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
            model="Unslothai 3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        if self.tokenizer is None:
            # unslothai/3 has no tokenizer files; use a compatible Llama 1/2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Unslothai 3 model from Hugging Face."""
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        # unslothai/3 config has all-zero dimensions which cause ZeroDivisionError
        # in LlamaConfig.__init__. Patch to minimal non-zero values and build from config.
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.update(
            {
                "head_dim": 1,
                "hidden_size": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 4,
                "vocab_size": self.tokenizer.vocab_size,
            }
        )
        config = LlamaConfig(**config_dict)

        model = LlamaModel(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
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
