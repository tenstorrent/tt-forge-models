# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-SW3 model loader implementation for causal language modeling (PyTorch).

The HuggingFace repo is gated, so models are constructed from config with
random weights for compile-only testing.
"""

from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel

_GPT_SW3_356M_CONFIG = {
    "vocab_size": 64000,
    "n_embd": 1024,
    "n_layer": 24,
    "n_head": 16,
    "n_positions": 2048,
    "activation_function": "gelu_new",
}


class ModelVariant(StrEnum):
    """Available GPT-SW3 model variants."""

    GPT_SW3_356M = "356M"


class ModelLoader(ForgeModel):
    """GPT-SW3 model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.GPT_SW3_356M: ModelConfig(
            pretrained_model_name="AI-Sweden-Models/gpt-sw3-356m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_SW3_356M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GPT-SW3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = GPT2Config(**_GPT_SW3_356M_CONFIG)
        model = GPT2LMHeadModel(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        seq_len = 16
        inputs = {
            "input_ids": torch.randint(
                0, _GPT_SW3_356M_CONFIG["vocab_size"], (1, seq_len)
            ),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        return inputs
