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
        import sys

        # vocos.feature_extractors imports `from encodec import EncodecModel` at module
        # level. The worktree root is in sys.path and the local encodec/ model directory
        # shadows the pip-installed encodec audio library. Reorder sys.path temporarily
        # to put site-packages first so vocos resolves the correct encodec package.
        original_path = sys.path.copy()
        site_pkgs = [p for p in sys.path if "site-packages" in p]
        other = [p for p in sys.path if "site-packages" not in p]
        sys.path[:] = site_pkgs + other
        try:
            from vocos import Vocos
            import torchaudio
            import vocos.feature_extractors as _vfe

            # vocos 0.1.0 MelSpectrogramFeatures lacks f_min/f_max/norm/mel_scale.
            # Patch it to accept and forward these to torchaudio.transforms.MelSpectrogram.
            class _PatchedMelSpectrogramFeatures(_vfe.FeatureExtractor):
                def __init__(
                    self,
                    sample_rate=24000,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=100,
                    padding="center",
                    f_min=0.0,
                    f_max=None,
                    norm=None,
                    mel_scale="htk",
                ):
                    super().__init__()
                    if padding not in ["center", "same"]:
                        raise ValueError("Padding must be 'center' or 'same'.")
                    self.padding = padding
                    self.mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                        center=padding == "center",
                        power=1,
                        f_min=f_min,
                        f_max=f_max,
                        norm=norm,
                        mel_scale=mel_scale,
                    )

                def forward(self, audio, **kwargs):
                    import torch
                    from vocos.feature_extractors import safe_log

                    if self.padding == "same":
                        pad = self.mel_spec.win_length - self.mel_spec.hop_length
                        audio = torch.nn.functional.pad(
                            audio, (pad // 2, pad // 2), mode="reflect"
                        )
                    mel = self.mel_spec(audio)
                    return safe_log(mel)

            _vfe.MelSpectrogramFeatures = _PatchedMelSpectrogramFeatures
        finally:
            sys.path[:] = original_path

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
