# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechT5 HiFiGAN Vocoder model loader implementation.

The vocoder converts spectrograms to audio waveforms.
"""
import torch
from transformers import SpeechT5HifiGan
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


class ModelVariant(StrEnum):
    """Available SpeechT5 HiFiGAN Vocoder model variants."""

    HIFIGAN = "hifigan"


class ModelLoader(ForgeModel):
    """SpeechT5 HiFiGAN Vocoder model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.HIFIGAN: ModelConfig(
            pretrained_model_name="microsoft/speecht5_hifigan",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HIFIGAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="speecht5_vocoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the SpeechT5 HiFiGAN Vocoder model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SpeechT5 HiFiGAN vocoder model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = SpeechT5HifiGan.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SpeechT5 HiFiGAN Vocoder model.

        The vocoder expects a spectrogram tensor with shape (batch, seq_len, mel_bins).
        - batch: batch size (1 for single inference)
        - seq_len: sequence length / number of frames (512)
        - mel_bins: number of mel frequency bins (80)

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's default dtype.

        Returns:
            torch.Tensor: Spectrogram tensor with shape (1, 512, 80).
        """
        # Create a random spectrogram tensor as sample input
        # Shape: (batch_size, sequence_length, num_mel_bins)
        spectrogram = torch.randn(1, 512, 80)

        if dtype_override is not None:
            spectrogram = spectrogram.to(dtype_override)

        return spectrogram
