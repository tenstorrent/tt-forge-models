# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
alVoCat Vocos 22kHz model loader implementation.

alVoCat is a Vocos-based neural vocoder for Catalan text-to-speech that
synthesizes 22 kHz audio waveforms from 80-bin mel-spectrograms.
"""
import torch
import torch.nn as nn
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


class VocosDecodeWrapper(nn.Module):
    """Wrapper around Vocos that exposes decode as the forward pass.

    The Vocos forward() method expects raw audio and re-encodes it internally.
    For mel-based inference, we only need backbone + head on the input features.
    """

    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.head = vocos.head

    def forward(self, features):
        x = self.backbone(features)
        audio = self.head(x)
        return audio


class ModelVariant(StrEnum):
    """Available alVoCat Vocos model variants."""

    ALVOCAT_22KHZ = "alvocat_22khz"


class ModelLoader(ForgeModel):
    """alVoCat Vocos 22kHz model loader implementation."""

    _VARIANTS = {
        ModelVariant.ALVOCAT_22KHZ: ModelConfig(
            pretrained_model_name="projecte-aina/alvocat-vocos-22khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALVOCAT_22KHZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="AlVoCatVocos",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the alVoCat Vocos model wrapped for decode inference.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: Wrapped Vocos model that decodes mel features to audio.
        """
        from vocos import Vocos

        pretrained_model_name = self._variant_config.pretrained_model_name

        vocos = Vocos.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            vocos = vocos.to(dtype=dtype_override)

        model = VocosDecodeWrapper(vocos)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the alVoCat Vocos model.

        Generates a random 80-bin mel-spectrogram shaped (batch, n_mels, frames)
        suitable for the Vocos decode backbone + head.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's default dtype.

        Returns:
            dict: Input tensors containing the mel-spectrogram features.
        """
        # Shape: (batch, n_mels=80, frames)
        features = torch.randn(1, 80, 256)

        if dtype_override is not None:
            features = features.to(dtype_override)

        return {"features": features}
