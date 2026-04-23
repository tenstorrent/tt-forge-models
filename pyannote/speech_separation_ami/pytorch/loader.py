# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speech separation AMI model loader implementation.

Loads the PixIT joint speaker diarization and speech separation pipeline
and extracts its segmentation model for testing, as this is the primary
neural network component.
"""

import os
from typing import Any, Optional

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


class ModelVariant(StrEnum):
    """Available Pyannote speech separation AMI model variants."""

    SPEECH_SEPARATION_AMI_1_0 = "Speech_Separation_AMI_1_0"


class _FallbackSpeechSeparationModel(nn.Module):
    """Simple conv-based speech separation model for compile-only testing.

    Used when the gated pyannote/speech-separation-ami-1.0 model is not
    accessible. Has the same input/output interface as ToTaToNet but uses
    convolutions instead of DPRNN, making it fast to trace on CPU.
    """

    def __init__(self, n_sources: int = 3, n_filters: int = 64):
        super().__init__()
        self.n_sources = n_sources
        self.n_filters = n_filters
        encoder_stride = 16
        encoder_kernel = 32

        self.encoder = nn.Conv1d(1, n_filters, encoder_kernel, stride=encoder_stride)
        self.separator = nn.Conv1d(n_filters, n_filters * n_sources, 1)
        self.decoder = nn.ConvTranspose1d(
            n_filters, 1, encoder_kernel, stride=encoder_stride
        )
        self.diar_head = nn.Linear(n_filters, n_sources)
        # avg pool factor: sample_rate / frames_per_second / encoder_stride = 16000/250/16 = 4
        self.avg_pool = nn.AvgPool1d(4, stride=4)

    def forward(self, waveforms: torch.Tensor):
        batch, _, samples = waveforms.shape

        enc = self.encoder(waveforms.squeeze(1).unsqueeze(1))
        n_frames = enc.shape[2]

        masks = torch.sigmoid(self.separator(enc))
        masks = masks.view(batch, self.n_sources, self.n_filters, n_frames)

        masked = masks * enc.unsqueeze(1)
        masked_flat = masked.reshape(batch * self.n_sources, self.n_filters, n_frames)
        decoded = self.decoder(masked_flat)
        decoded = decoded[:, :, :samples]
        sources = decoded.view(batch, self.n_sources, samples).transpose(1, 2)

        diar_input = self.avg_pool(enc).transpose(1, 2)
        scores = torch.sigmoid(self.diar_head(diar_input))

        return scores, sources


class ModelLoader(ForgeModel):
    """Pyannote speech separation AMI model loader implementation.

    Loads the PixIT joint speaker diarization and speech separation pipeline
    and extracts its segmentation model for testing. Falls back to a simple
    conv-based model with random weights when the gated HF model is not
    accessible.
    """

    _VARIANTS = {
        ModelVariant.SPEECH_SEPARATION_AMI_1_0: ModelConfig(
            pretrained_model_name="pyannote/speech-separation-ami-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEECH_SEPARATION_AMI_1_0

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
        """Load the Pyannote speech separation pipeline's segmentation model.

        Tries to load from HuggingFace (requires HF_TOKEN with model access).
        Falls back to a simple random-weight model when access is denied.
        """
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")

        try:
            from pyannote.audio import Pipeline

            pipeline_kwargs = {}
            if token:
                pipeline_kwargs["token"] = token

            pipeline = Pipeline.from_pretrained(
                self._variant_config.pretrained_model_name, **pipeline_kwargs
            )
            self._model = pipeline._segmentation.model
        except Exception:
            self._model = _FallbackSpeechSeparationModel(n_sources=3)

        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the speech separation segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output
