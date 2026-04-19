# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Arabic model loader implementation for speech recognition (ASR).

Byne/whisper-large-v3-arabic is a fine-tuned version of openai/whisper-large-v3
for Arabic language automatic speech recognition.
"""

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from typing import Optional

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
    """Available Whisper Arabic model variants."""

    LARGE_V3_ARABIC = "Large_v3_Arabic"


class ModelLoader(ForgeModel):
    """Whisper Arabic model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_ARABIC: ModelConfig(
            pretrained_model_name="Byne/whisper-large-v3-arabic",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_ARABIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Arabic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Whisper Arabic model from Hugging Face."""
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
        """Generate sample inputs for Whisper Arabic model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio (30 seconds at 16kHz)
        sample_audio = np.random.randn(16000 * 30).astype(np.float32)
        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=16000
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
