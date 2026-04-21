# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice-ASR model loader implementation for speech recognition (ASR).

VibeVoice-ASR is a speech model from Microsoft for automatic speech recognition,
transcription, and speaker diarization. This loader uses the official HuggingFace
variant (microsoft/VibeVoice-ASR-HF) with native transformers support.
"""

from typing import Optional

import numpy as np
import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available VibeVoice-ASR speech recognition model variants."""

    V4BIT = "4bit"


class ModelLoader(ForgeModel):
    """VibeVoice-ASR model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.V4BIT: ModelConfig(
            pretrained_model_name="microsoft/VibeVoice-ASR-HF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VibeVoice_ASR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the audio processor for the model."""
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VibeVoice-ASR model instance."""
        from transformers import AutoModelForSpeechSeq2Seq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VibeVoice-ASR model."""
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        sampling_rate = 24000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            text="transcribe",
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if (
                    isinstance(inputs[key], torch.Tensor)
                    and inputs[key].is_floating_point()
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
