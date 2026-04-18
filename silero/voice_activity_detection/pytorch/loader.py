# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Silero VAD (Voice Activity Detection) model loader implementation.

Loads the Silero VAD v5 model via torch.hub from snakers4/silero-vad.
The model detects speech segments in audio, returning a probability
that a given audio chunk contains speech.
"""

from typing import Optional

import torch
import torch.nn as nn

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


class SileroVADWrapper(nn.Module):
    """Wraps the Silero VAD JIT model for compatibility with nn.Module conventions.

    The JIT ScriptModule has plain Tensor parameters (not nn.Parameter) and
    mutable internal state, both incompatible with nn.Module._apply() in
    PyTorch 2.9+. We store the JIT model outside the submodule registry
    and skip device/dtype movement — torch.compile handles device placement
    during tracing.
    """

    def __init__(self, jit_model):
        super().__init__()
        object.__setattr__(self, "_jit", jit_model)

    def forward(self, x, sr):
        return self._jit(x, sr)


class ModelVariant(StrEnum):
    """Available Silero VAD model variants."""

    SILERO_VAD_V5 = "Silero_VAD_v5"


class ModelLoader(ForgeModel):
    """Silero VAD model loader implementation for voice activity detection (PyTorch)."""

    _VARIANTS = {
        ModelVariant.SILERO_VAD_V5: ModelConfig(
            pretrained_model_name="snakers4/silero-vad",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SILERO_VAD_V5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SileroVAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        jit_model, _ = torch.hub.load(
            repo_or_dir=self._variant_config.pretrained_model_name,
            model="silero_vad",
            trust_repo=True,
        )

        model = SileroVADWrapper(jit_model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        chunk_size = 512
        sampling_rate = 16000

        audio_chunk = torch.randn(1, chunk_size)

        return [audio_chunk, sampling_rate]
