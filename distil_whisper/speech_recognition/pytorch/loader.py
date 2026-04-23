# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Distil-Whisper model loader implementation for speech recognition (ASR) using PyTorch.

Distil-small.en is a distilled version of OpenAI's Whisper small.en,
6x faster with 49% fewer parameters while performing within 1% WER.

The distil-large-v3.5-ct2 variant is a CTranslate2-quantized conversion of
distil-whisper/distil-large-v3.5. Since the CTranslate2 format is not
compatible with PyTorch, this loader uses the base distil-whisper/distil-large-v3.5
weights via AutoModelForSpeechSeq2Seq.
"""

from typing import Optional

import numpy as np

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
    """Available Distil-Whisper PyTorch speech recognition model variants."""

    DISTIL_SMALL_EN = "Distil_small_en"
    DISTIL_LARGE_V3_5_CT2 = "Distil_large_v3_5_ct2"


class ModelLoader(ForgeModel):
    """Distil-Whisper model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.DISTIL_SMALL_EN: ModelConfig(
            pretrained_model_name="distil-whisper/distil-small.en",
        ),
        ModelVariant.DISTIL_LARGE_V3_5_CT2: ModelConfig(
            pretrained_model_name="distil-whisper/distil-large-v3.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTIL_SMALL_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Distil_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 1-second audio waveform at 16kHz
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

        # Whisper is encoder-decoder; provide decoder_input_ids with start token
        decoder_start_token_id = self._processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        inputs["decoder_input_ids"] = torch.tensor(
            [[decoder_start_token_id]], dtype=torch.long
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v
                for k, v in inputs.items()
            }

        return inputs
