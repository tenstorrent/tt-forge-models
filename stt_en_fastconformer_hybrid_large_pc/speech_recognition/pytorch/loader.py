# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
STT EN FastConformer Hybrid Large PC model loader for speech recognition (ASR).

nvidia/stt_en_fastconformer_hybrid_large_pc is a large (~114M parameter) NeMo
FastConformer Hybrid RNNT-CTC model trained for English automatic speech
recognition with punctuation and capitalization.
"""

import torch
from typing import Optional

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
    """Available STT EN FastConformer Hybrid Large PC model variants."""

    LARGE_PC = "large_pc"


class ModelLoader(ForgeModel):
    """STT EN FastConformer Hybrid Large PC model loader for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_PC: ModelConfig(
            pretrained_model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="STT_EN_FastConformer_Hybrid",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

        model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        input_signal = torch.tensor(audio_array).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio_array)])

        if dtype_override is not None:
            input_signal = input_signal.to(dtype_override)

        return {
            "input_signal": input_signal,
            "input_signal_length": input_signal_length,
        }
