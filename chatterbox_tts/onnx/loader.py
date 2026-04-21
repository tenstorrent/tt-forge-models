# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chatterbox Turbo ONNX model loader implementation for text-to-speech tasks.
"""

from typing import Optional

import onnx
import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Chatterbox Turbo ONNX model variants."""

    CHATTERBOX_TURBO = "chatterbox-turbo"


class ModelLoader(ForgeModel):
    """Chatterbox Turbo ONNX model loader for text-to-speech tasks.

    Loads the language model component (a GPT-2 style backbone) from the
    ResembleAI/chatterbox-turbo-ONNX repository. The language model produces
    speech token logits from input embeddings and is the primary neural
    network used during autoregressive generation.
    """

    _VARIANTS = {
        ModelVariant.CHATTERBOX_TURBO: ModelConfig(
            pretrained_model_name="ResembleAI/chatterbox-turbo-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHATTERBOX_TURBO

    # Architecture constants for the language model backbone.
    NUM_LAYERS = 24
    NUM_KV_HEADS = 16
    HEAD_DIM = 64
    HIDDEN_SIZE = 1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Chatterbox Turbo ONNX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and return the Chatterbox Turbo language model ONNX file."""
        from huggingface_hub import hf_hub_download

        repo_id = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(repo_id=repo_id, filename="onnx/language_model.onnx")
        # External weights file must live next to the main ONNX file.
        hf_hub_download(repo_id=repo_id, filename="onnx/language_model.onnx_data")

        return onnx.load(model_path)

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Chatterbox Turbo language model.

        The language model expects input embeddings plus attention/position
        indices and an empty KV cache entry per transformer layer. This
        matches the first step of autoregressive generation.
        """
        batch_size = 1
        seq_len = 32

        inputs_embeds = torch.randn(
            batch_size, seq_len, self.HIDDEN_SIZE, dtype=torch.float32
        )
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
        position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

        past_shape = (batch_size, self.NUM_KV_HEADS, 0, self.HEAD_DIM)
        past_key_values = {}
        for layer in range(self.NUM_LAYERS):
            past_key_values[f"past_key_values.{layer}.key"] = torch.zeros(
                past_shape, dtype=torch.float32
            )
            past_key_values[f"past_key_values.{layer}.value"] = torch.zeros(
                past_shape, dtype=torch.float32
            )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past_key_values,
        }
