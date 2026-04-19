# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MedASR model loader implementation for speech recognition (ASR) using PyTorch.

The google/medasr repo is gated, so the model is created from a Wav2Vec2
config with random weights for compile-only testing.
"""

from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MedASR PyTorch speech recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """MedASR model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="google/medasr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MedASR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

        config = Wav2Vec2Config(
            vocab_size=32,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        model = Wav2Vec2ForCTC(config)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.float32
        sampling_rate = 16000
        duration_seconds = 1
        input_values = torch.randn(1, sampling_rate * duration_seconds, dtype=dtype)

        return {"input_values": input_values}
