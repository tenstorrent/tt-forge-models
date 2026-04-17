# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni AudioTransformer model loader implementation for audio feature extraction.

This model is the audio encoder component extracted from Qwen3-Omni-30B-A3B-Instruct.
It is a Whisper-style transformer encoder that produces audio embeddings from raw
audio waveforms.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3-Omni AudioTransformer model variants."""

    DEFAULT = "default"


class Qwen3OmniAudioEncoderWrapper(nn.Module):
    """Wrapper that calls encoder submodules directly with static shapes.

    The original forward uses data-dependent split/pad operations that are
    incompatible with torch.compile. This wrapper pre-chunks the mel features
    into fixed-size windows and runs convolution, transformer, and projection
    layers with deterministic tensor shapes.
    """

    def __init__(self, model):
        super().__init__()
        self.conv2d1 = model.conv2d1
        self.conv2d2 = model.conv2d2
        self.conv2d3 = model.conv2d3
        self.conv_out = model.conv_out
        self.positional_embedding = model.positional_embedding
        self.layers = model.layers
        self.ln_post = model.ln_post
        self.proj1 = model.proj1
        self.act = model.act
        self.proj2 = model.proj2
        self.num_heads = model.layers[0].self_attn.num_heads
        self.head_dim = model.layers[0].self_attn.head_dim
        self.scaling = model.layers[0].self_attn.scaling

    def forward(self, padded_input):
        # padded_input: (num_chunks, 1, mel_bins, chunk_time)
        x = F.gelu(self.conv2d1(padded_input))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        pos = (
            self.positional_embedding.positional_embedding[:t, :]
            .unsqueeze(0)
            .to(x.dtype)
        )
        x = x + pos
        # (num_chunks, t, embed_dim) -> (num_chunks * t, embed_dim)
        x = x.reshape(-1, x.shape[-1])

        for layer in self.layers:
            residual = x
            x = layer.self_attn_layer_norm(x)
            seq_len = x.shape[0]
            q = self.scaling * layer.self_attn.q_proj(x).reshape(
                seq_len, self.num_heads, self.head_dim
            )
            k = layer.self_attn.k_proj(x).reshape(
                seq_len, self.num_heads, self.head_dim
            )
            v = layer.self_attn.v_proj(x).reshape(
                seq_len, self.num_heads, self.head_dim
            )
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn_out = (
                attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, -1).contiguous()
            )
            x = residual + layer.self_attn.out_proj(attn_out)
            residual = x
            x = layer.final_layer_norm(x)
            x = layer.fc1(x)
            x = layer.activation_fn(x)
            x = layer.fc2(x)
            x = residual + x

        x = self.ln_post(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class ModelLoader(ForgeModel):
    """Qwen3-Omni AudioTransformer model loader for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Atotti/Qwen3-Omni-AudioTransformer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3_Omni_AudioTransformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import WhisperFeatureExtractor

        self._processor = WhisperFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-Omni AudioTransformer model."""
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            Qwen3OmniMoeAudioEncoderConfig,
        )

        config = Qwen3OmniMoeAudioEncoderConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen3OmniMoeAudioEncoder.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return Qwen3OmniAudioEncoderWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen3-Omni AudioTransformer."""
        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # (1, mel_bins, time) -> (mel_bins, time)
        input_features = inputs["input_features"].squeeze(0)
        if dtype_override is not None:
            input_features = input_features.to(dtype_override)

        mel_bins, time_steps = input_features.shape
        n_window = 50
        chunk_size = n_window * 2  # 100
        num_chunks = (time_steps + chunk_size - 1) // chunk_size
        pad_len = num_chunks * chunk_size - time_steps
        if pad_len > 0:
            input_features = F.pad(input_features, (0, pad_len))

        # (mel_bins, time) -> (num_chunks, mel_bins, chunk_size) -> (num_chunks, 1, mel_bins, chunk_size)
        padded_input = (
            input_features.reshape(mel_bins, num_chunks, chunk_size)
            .permute(1, 0, 2)
            .unsqueeze(1)
        )

        return [padded_input]
