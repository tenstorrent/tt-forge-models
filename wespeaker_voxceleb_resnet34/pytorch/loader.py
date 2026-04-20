# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WeSpeaker VoxCeleb ResNet34-LM model loader implementation for speaker embeddings.
"""

from typing import Optional

import torch
import torch.nn as nn

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class WeSpeakerResNet34Wrapper(nn.Module):
    """Wrapper that keeps the model in float32 for FFT-based fbank computation."""

    def __init__(self, model, output_dtype=None):
        super().__init__()
        self.model = model
        self.output_dtype = output_dtype

    def forward(self, waveforms):
        out = self.model(waveforms.float())
        if self.output_dtype is not None:
            out = out.to(self.output_dtype)
        return out


class ModelVariant(StrEnum):
    """Available WeSpeaker VoxCeleb ResNet34-LM model variants."""

    RESNET34_LM = "ResNet34_LM"


class ModelLoader(ForgeModel):
    """WeSpeaker VoxCeleb ResNet34-LM model loader implementation."""

    _VARIANTS = {
        ModelVariant.RESNET34_LM: ModelConfig(
            pretrained_model_name="pyannote/wespeaker-voxceleb-resnet34-LM",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RESNET34_LM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WeSpeakerVoxCelebResNet34",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the WeSpeaker VoxCeleb ResNet34-LM model from pyannote.audio."""
        from pyannote.audio import Model

        model = Model.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()

        wrapped = WeSpeakerResNet34Wrapper(model, output_dtype=dtype_override)
        wrapped.eval()

        self.model = wrapped
        return wrapped

    def load_inputs(self, dtype_override=None):
        """Generate sample audio input for the model.

        Returns a 1-second mono waveform at 16kHz sample rate.
        """
        waveform = torch.randn(1, 1, 16000)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return [waveform]
