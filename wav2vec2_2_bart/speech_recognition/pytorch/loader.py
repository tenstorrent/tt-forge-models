# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wav2Vec2-to-BART model loader implementation for speech recognition (ASR) using PyTorch.

The model is a SpeechEncoderDecoderModel combining a wav2vec2-base encoder with a
bart-base decoder, fine-tuned on the LibriSpeech clean dataset.
"""

from typing import Optional

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
    """Available Wav2Vec2-to-BART PyTorch speech recognition model variants."""

    BART_BASE = "patrickvonplaten/wav2vec2-2-bart-base"


class ModelLoader(ForgeModel):
    """Wav2Vec2-to-BART model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BART_BASE: ModelConfig(
            pretrained_model_name="patrickvonplaten/wav2vec2-2-bart-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BART_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Vec2-2-BART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import SpeechEncoderDecoderModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = SpeechEncoderDecoderModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor()
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

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
            inputs["input_values"] = inputs["input_values"].to(dtype_override)

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
