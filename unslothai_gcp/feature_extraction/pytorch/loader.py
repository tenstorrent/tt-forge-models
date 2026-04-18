# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unslothai GCP model loader implementation for feature extraction.

The unslothai/gcp HuggingFace repo is a health-check placeholder with
zero-dimension config and no tokenizer files. We instantiate a minimal
LlamaModel from config and generate dummy token inputs directly.
"""
import torch
from transformers import LlamaConfig, LlamaModel
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

_VOCAB_SIZE = 32000
_HIDDEN_SIZE = 64
_NUM_HEADS = 2
_NUM_LAYERS = 2
_INTERMEDIATE_SIZE = 128
_MAX_POSITION_EMBEDDINGS = 128


class ModelVariant(StrEnum):
    """Available Unslothai GCP model variants for feature extraction."""

    UNSLOTHAI_GCP = "unslothai/gcp"


class ModelLoader(ForgeModel):
    """Unslothai GCP model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.UNSLOTHAI_GCP: LLMModelConfig(
            pretrained_model_name="unslothai/gcp",
            max_length=32,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTHAI_GCP

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Unslothai GCP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Create a minimal LlamaModel since the HF repo has zero-dim config."""
        config = LlamaConfig(
            vocab_size=_VOCAB_SIZE,
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_attention_heads=_NUM_HEADS,
            num_hidden_layers=_NUM_LAYERS,
            num_key_value_heads=_NUM_HEADS,
            max_position_embeddings=_MAX_POSITION_EMBEDDINGS,
        )
        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model = LlamaModel(config)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None, query=None):
        """Generate dummy token inputs (no tokenizer available in this repo)."""
        max_length = self._variant_config.max_length
        input_ids = torch.randint(0, _VOCAB_SIZE, (1, max_length))
        attention_mask = torch.ones(1, max_length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

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
