# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Silero VAD v5 (MLX) model loader implementation for voice activity detection.
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


class SileroVADv5(nn.Module):
    """PyTorch implementation of Silero VAD v5 architecture.

    Architecture: STFT -> Conv1d encoder -> LSTM -> Conv1d decoder -> sigmoid
    """

    def __init__(
        self,
        filter_length=256,
        hop_length=128,
        encoder_channels=None,
        encoder_kernel_sizes=None,
        encoder_strides=None,
        lstm_hidden_size=128,
        lstm_num_layers=1,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [129, 128, 64, 64, 128]
        if encoder_kernel_sizes is None:
            encoder_kernel_sizes = [3, 3, 3, 3]
        if encoder_strides is None:
            encoder_strides = [1, 2, 2, 1]

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.lstm_hidden_size = lstm_hidden_size

        # Build encoder: sequence of Conv1d + ReLU layers
        encoder_layers = []
        for i in range(len(encoder_kernel_sizes)):
            encoder_layers.append(
                nn.Conv1d(
                    encoder_channels[i],
                    encoder_channels[i + 1],
                    kernel_size=encoder_kernel_sizes[i],
                    stride=encoder_strides[i],
                    padding=encoder_kernel_sizes[i] // 2,
                )
            )
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=encoder_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Decoder: single Conv1d to produce speech probability
        self.decoder = nn.Conv1d(lstm_hidden_size, 1, kernel_size=1)

    def forward(self, audio_chunk):
        """Process an audio chunk and return speech probability.

        Args:
            audio_chunk: Tensor of shape (batch, chunk_size)

        Returns:
            Tensor of shape (batch, 1) with speech probability
        """
        # Compute STFT magnitude (always float32, cast to model dtype after)
        stft = torch.stft(
            audio_chunk.float(),
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.filter_length,
            window=torch.hann_window(self.filter_length, device=audio_chunk.device),
            return_complex=True,
        )
        x = stft.abs().to(self.encoder[0].weight.dtype)

        # Encoder
        x = self.encoder(x)  # (batch, channels, time_frames)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        # Decoder
        x = self.decoder(x)  # (batch, 1, time_frames)

        # Average over time and apply sigmoid
        x = x.mean(dim=-1)  # (batch, 1)
        x = torch.sigmoid(x)

        return x


class ModelVariant(StrEnum):
    """Available Silero VAD v5 model variants."""

    SILERO_VAD_V5_MLX = "Silero_VAD_v5_MLX"


class ModelLoader(ForgeModel):
    """Silero VAD v5 (MLX) model loader for voice activity detection."""

    _VARIANTS = {
        ModelVariant.SILERO_VAD_V5_MLX: ModelConfig(
            pretrained_model_name="aufklarer/Silero-VAD-v5-MLX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SILERO_VAD_V5_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Silero-VAD-v5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _convert_mlx_state_dict(mlx_sd):
        """Convert MLX-format state dict to PyTorch format."""
        pt_sd = {}

        # Encoder: MLX indices 0-3 map to PyTorch indices 0,2,4,6 (ReLU at 1,3,5,7)
        for mlx_idx in range(4):
            pt_idx = mlx_idx * 2
            w = mlx_sd[f"encoder.{mlx_idx}.weight"]
            # MLX Conv1d: (out, kernel, in) -> PyTorch: (out, in, kernel)
            pt_sd[f"encoder.{pt_idx}.weight"] = w.permute(0, 2, 1)
            pt_sd[f"encoder.{pt_idx}.bias"] = mlx_sd[f"encoder.{mlx_idx}.bias"]

        # LSTM: MLX uses Wx/Wh/bias, PyTorch uses weight_ih/weight_hh/bias_ih/bias_hh
        pt_sd["lstm.weight_ih_l0"] = mlx_sd["lstm.Wx"]
        pt_sd["lstm.weight_hh_l0"] = mlx_sd["lstm.Wh"]
        pt_sd["lstm.bias_ih_l0"] = mlx_sd["lstm.bias"]
        pt_sd["lstm.bias_hh_l0"] = torch.zeros_like(mlx_sd["lstm.bias"])

        # Decoder Conv1d: same transposition
        pt_sd["decoder.weight"] = mlx_sd["decoder.weight"].permute(0, 2, 1)
        pt_sd["decoder.bias"] = mlx_sd["decoder.bias"]

        return pt_sd

    def load_model(self, *, dtype_override=None, **kwargs):
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        model = SileroVADv5()

        weights_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.safetensors",
        )
        mlx_sd = load_file(weights_path)
        pt_sd = self._convert_mlx_state_dict(mlx_sd)
        model.load_state_dict(pt_sd, strict=True)
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Generate a synthetic audio chunk: 512 samples at 16kHz (32ms)
        chunk_size = 512
        audio_chunk = torch.randn(1, chunk_size)

        if dtype_override is not None:
            audio_chunk = audio_chunk.to(dtype_override)

        return (audio_chunk,)
