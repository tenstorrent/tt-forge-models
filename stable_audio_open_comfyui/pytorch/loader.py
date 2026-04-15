#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio Open 1.0 model loader implementation.

Loads the StableAudioDiTModel transformer component for testing.
In compile-only mode (TT_RANDOM_WEIGHTS=1), creates the model with random weights.
Otherwise, loads from the stabilityai/stable-audio-open-1.0 pretrained pipeline.

Available variants:
- STABLE_AUDIO_OPEN_1_0: Stable Audio Open 1.0 DiT transformer
"""

import os
from typing import Any, Optional

import torch
from diffusers import StableAudioDiTModel  # type: ignore[import]

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "stabilityai/stable-audio-open-1.0"

# DiT model hidden size (from Stable Audio Open 1.0 config)
HIDDEN_SIZE = 1024


class ModelVariant(StrEnum):
    """Available Stable Audio Open model variants."""

    STABLE_AUDIO_OPEN_1_0 = "1.0"


class ModelLoader(ForgeModel):
    """Stable Audio Open 1.0 model loader for the DiT transformer component."""

    _VARIANTS = {
        ModelVariant.STABLE_AUDIO_OPEN_1_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.STABLE_AUDIO_OPEN_1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="STABLE_AUDIO_OPEN_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Stable Audio DiT transformer model.

        Returns:
            StableAudioDiTModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self._transformer = StableAudioDiTModel()
            if dtype_override is not None:
                self._transformer = self._transformer.to(dtype=dtype_override)
        else:
            from diffusers import StableAudioPipeline  # type: ignore[import]

            pipeline = StableAudioPipeline.from_pretrained(
                REPO_ID,
                torch_dtype=dtype,
            )
            self._transformer = pipeline.transformer

        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the StableAudioDiTModel transformer.

        Returns a dictionary of tensors matching the transformer's forward signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)
        seq_length = kwargs.get("seq_length", 8)

        hidden_states = torch.randn(batch_size, seq_length, HIDDEN_SIZE, dtype=dtype)
        timestep = torch.rand(batch_size, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, seq_length, HIDDEN_SIZE, dtype=dtype
        )
        global_hidden_states = torch.randn(batch_size, 1, HIDDEN_SIZE, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "global_hidden_states": global_hidden_states,
        }
