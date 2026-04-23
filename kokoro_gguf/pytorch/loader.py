# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro GGUF model loader implementation for text-to-speech tasks.

Loads the Kokoro TTS model from hexgrad/Kokoro-82M via the kokoro package.
The model's Decoder component is used for compilation since it accepts
fixed-size tensor inputs (asr, F0_curve, N, style_vector).
"""
import torch
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

REPO_ID = "hexgrad/Kokoro-82M"

# Decoder input dimensions from kokoro config
_HIDDEN_DIM = 512
_STYLE_DIM = 128
# asr sequence length; F0/N must be 2x due to stride-2 conv in decoder
_ASR_SEQ_LEN = 50
_MEL_SEQ_LEN = _ASR_SEQ_LEN * 2


class ModelVariant(StrEnum):
    """Available Kokoro GGUF model variants."""

    KOKORO_NO_ESPEAK_Q4_GGUF = "no_espeak_Q4_GGUF"


class ModelLoader(ForgeModel):
    """Kokoro GGUF model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KOKORO_NO_ESPEAK_Q4_GGUF: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOKORO_NO_ESPEAK_Q4_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None, **kwargs):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kokoro GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from kokoro import KModel

        full_model = KModel(repo_id=REPO_ID)
        # Keep float32: the sinusoidal source module generates float32 tensors
        # internally, so bfloat16 casting causes dtype mismatches at runtime.
        model = full_model.decoder.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = torch.float32
        return {
            "asr": torch.randn(batch_size, _HIDDEN_DIM, _ASR_SEQ_LEN, dtype=dtype),
            "F0_curve": torch.randn(batch_size, _MEL_SEQ_LEN, dtype=dtype),
            "N": torch.randn(batch_size, _MEL_SEQ_LEN, dtype=dtype),
            "s": torch.randn(batch_size, _STYLE_DIM, dtype=dtype),
        }

    def load_config(self):
        return self._variant_config
