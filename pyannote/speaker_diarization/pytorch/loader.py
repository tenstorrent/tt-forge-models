# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speaker diarization model loader implementation.

Implements the PyanNet segmentation model architecture (SincNet + BiLSTM +
linear projection + classifier) used as the backbone in
pyannote/speaker-diarization-3.x, using only torch.nn primitives.

This avoids the pyannote.audio / torchaudio dependency chain (which requires
CUDA-linked torchaudio and forces a numpy upgrade that breaks in-process) while
exercising the same compiler patterns: Conv1d, MaxPool1d, InstanceNorm1d, LSTM,
Linear, Sigmoid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
    """Available Pyannote speaker diarization model variants."""

    DIARIZATION_3_0 = "Diarization_3_0"
    DIARIZATION_3_1 = "Diarization_3_1"


class _SincNetBlock(nn.Module):
    """Synthetic SincNet frontend matching pyannote/segmentation-3.0 architecture.

    Replicates the three conv+pool+norm stages of SincNet (stride=10) using
    standard Conv1d instead of the learnable sinc filterbank from
    asteroid-filterbanks.  Output shape is identical.
    """

    def __init__(self, stride: int = 10):
        super().__init__()
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        # Stage 1: replaces Encoder(ParamSincFB(80, 251, stride=stride))
        self.conv1 = nn.Conv1d(1, 80, kernel_size=251, stride=stride, padding=0)
        self.pool1 = nn.MaxPool1d(3, stride=3)
        self.norm1 = nn.InstanceNorm1d(80, affine=True)
        # Stage 2
        self.conv2 = nn.Conv1d(80, 60, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool1d(3, stride=3)
        self.norm2 = nn.InstanceNorm1d(60, affine=True)
        # Stage 3
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, stride=1)
        self.pool3 = nn.MaxPool1d(3, stride=3)
        self.norm3 = nn.InstanceNorm1d(60, affine=True)

    def forward(self, x):
        x = self.wav_norm1d(x)
        x = F.leaky_relu(self.norm1(self.pool1(self.conv1(x))))
        x = F.leaky_relu(self.norm2(self.pool2(self.conv2(x))))
        x = F.leaky_relu(self.norm3(self.pool3(self.conv3(x))))
        return x


class _PyanNet(nn.Module):
    """PyanNet: SincNet > BiLSTM > Feed-forward > Classifier.

    Architecture matches pyannote/segmentation-3.0:
      - 60-feature SincNet frontend (stride=10)
      - 4-layer bidirectional LSTM (hidden_size=128)
      - 2 linear layers (hidden_size=128, leaky_relu)
      - 7-class sigmoid output
    """

    def __init__(self):
        super().__init__()
        self.sincnet = _SincNetBlock(stride=10)
        self.lstm = nn.LSTM(
            input_size=60,
            hidden_size=128,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
            dropout=0.5,
        )
        self.linear = nn.ModuleList([
            nn.Linear(256, 128),
            nn.Linear(128, 128),
        ])
        self.classifier = nn.Linear(128, 7)
        self.activation = nn.Sigmoid()

    def forward(self, waveforms):
        outputs = self.sincnet(waveforms)
        outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
        outputs, _ = self.lstm(outputs)
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))
        return self.activation(self.classifier(outputs))


class ModelLoader(ForgeModel):
    """Pyannote speaker diarization model loader.

    Instantiates the PyanNet segmentation model (backbone of
    pyannote/speaker-diarization-3.x) with random weights, using only
    torch.nn primitives to avoid the pyannote.audio / torchaudio dependency
    chain.
    """

    _VARIANTS = {
        ModelVariant.DIARIZATION_3_0: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.0",
        ),
        ModelVariant.DIARIZATION_3_1: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIARIZATION_3_1

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pyannote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Instantiate PyanNet with random weights."""
        self._model = _PyanNet()
        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Return a 10-second mono 16kHz waveform: (1, 1, 160000)."""
        dtype = dtype_override or torch.float32
        return [torch.randn(1, 1, 160000, dtype=dtype)]
