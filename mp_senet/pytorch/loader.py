# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MP-SENet speech enhancement model loader implementation.

MP-SENet performs magnitude and phase speech enhancement in parallel,
denoising audio waveforms using a conformer-based architecture.
"""

import torch
import torch.nn as nn
import numpy as np
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

# STFT hyperparameters for the DNS checkpoint (from JacobLinCool/MP-SENet-DNS config)
_DNS_N_FFT = 400
_DNS_HOP_SIZE = 100
_DNS_WIN_SIZE = 400
_DNS_COMPRESS_FACTOR = 0.3
_DNS_SEGMENT_SIZE = 32000


class ModelVariant(StrEnum):
    """Available MP-SENet model variants."""

    DNS = "DNS"


class _MPSENetForwardWrapper(nn.Module):
    """Wraps MPSENet to expose forward(noisy_amp, noisy_pha) through __call__.

    MPSENet.__call__ is overridden to do STFT preprocessing internally,
    which requires float32 and cannot run on TT device.  This wrapper
    bypasses __call__ and invokes the pure-neural-network forward() path
    directly, so the compiled graph never sees complex-valued STFT tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_amp, noisy_pha):
        return self.model.forward(noisy_amp, noisy_pha)


class ModelLoader(ForgeModel):
    """MP-SENet speech enhancement model loader implementation."""

    _VARIANTS = {
        ModelVariant.DNS: ModelConfig(
            pretrained_model_name="JacobLinCool/MP-SENet-DNS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DNS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MP-SENet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from MPSENet import MPSENet

        model = MPSENet.from_pretrained(self._variant_config.pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return _MPSENetForwardWrapper(model)

    def load_inputs(self, dtype_override=None):
        # Pre-compute STFT on CPU in float32.  MKL FFT doesn't support bfloat16,
        # and torch.stft returns a complex tensor that TT XLA cannot handle.
        # model.forward() takes real-valued magnitude/phase spectra directly.
        from MPSENet.model.mpsenet import mag_pha_stft

        hann_window = torch.hann_window(_DNS_WIN_SIZE)
        audio = np.random.randn(_DNS_SEGMENT_SIZE).astype(np.float32)
        segment = torch.from_numpy(audio).unsqueeze(0)

        noisy_amp, noisy_pha, _ = mag_pha_stft(
            segment,
            hann_window,
            _DNS_N_FFT,
            _DNS_HOP_SIZE,
            _DNS_WIN_SIZE,
            _DNS_COMPRESS_FACTOR,
        )

        if dtype_override is not None:
            noisy_amp = noisy_amp.to(dtype_override)
            noisy_pha = noisy_pha.to(dtype_override)

        return noisy_amp, noisy_pha
