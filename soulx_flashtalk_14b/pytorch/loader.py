#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SoulX-FlashTalk-14B audio encoder loader implementation.

SoulX-FlashTalk-14B (Soul-AILab/SoulX-FlashTalk-14B) is a real-time streaming
audio-driven avatar animation model. It is built on top of InfiniteTalk and
Wan2.1-I2V-14B, using self-correcting bidirectional distillation to produce
infinite streaming video generation from a portrait image and audio.

This loader tests the Wav2Vec2 audio encoder component
(TencentGameMate/chinese-wav2vec2-base) used for audio conditioning in the
SoulX-FlashTalk pipeline.
"""

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available SoulX-FlashTalk-14B model variants."""

    SINGLE = "single"


class ModelLoader(ForgeModel):
    """SoulX-FlashTalk-14B audio encoder loader for audio-driven avatar animation."""

    _VARIANTS = {
        ModelVariant.SINGLE: ModelConfig(
            pretrained_model_name="TencentGameMate/chinese-wav2vec2-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SINGLE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize SoulX-FlashTalk-14B audio encoder loader."""
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SoulX-FlashTalk-14B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the Wav2Vec2 feature extractor for audio preprocessing."""
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Wav2Vec2 audio encoder used by SoulX-FlashTalk-14B."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Wav2Vec2Model.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        if self._processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return synthetic audio inputs for the audio encoder."""
        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return dict(inputs)
