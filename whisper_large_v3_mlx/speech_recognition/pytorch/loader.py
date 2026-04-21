# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large V3 MLX model loader implementation for speech recognition (ASR).

mlx-community/whisper-large-v3-mlx is an MLX conversion of openai/whisper-large-v3
for automatic speech recognition. Since the MLX format (weights.npz) is not directly
loadable by PyTorch, this loader uses the original openai/whisper-large-v3 weights
which are numerically identical.
"""

from typing import Optional

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Whisper Large V3 MLX model variants."""

    LARGE_V3_MLX = "Large_v3_MLX"


class ModelLoader(ForgeModel):
    """Whisper Large V3 MLX model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_MLX: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Large_V3_MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Whisper Large V3 model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Whisper Large V3 MLX model."""
        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
