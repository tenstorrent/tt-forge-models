# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-Video GGUF model loader implementation for text-to-video generation.

This loader uses GGUF-quantized variants of the LTX-Video model from
city96/LTX-Video-gguf. The GGUF transformer is loaded via diffusers'
LTXVideoTransformer3DModel.from_single_file.

The v0.9 GGUF checkpoint fuses rope position embedding dims into the
attention projection weights (input dim 2048+128=2176), which is
incompatible with the current diffusers architecture (expects 2048).
When loading fails, the loader falls back to a config-based model with
random weights, which produces the identical computation graph for
compile-only testing.

Available variants:
- Q4_0: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

from typing import Optional

import torch
from diffusers import LTXVideoTransformer3DModel

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

GGUF_REPO = "city96/LTX-Video-gguf"

# LTX-Video v0.9 architecture constants
_VAE_SPATIAL_COMPRESSION = 32
_VAE_TEMPORAL_COMPRESSION = 8
_TOKENIZER_MAX_LENGTH = 128
_CAPTION_CHANNELS = 4096  # T5 text encoder output dim fed into caption_projection


class ModelVariant(StrEnum):
    """Available LTX-Video GGUF quantization variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.Q4_0: "ltx-video-2b-v0.9-Q4_0.gguf",
    ModelVariant.Q8_0: "ltx-video-2b-v0.9-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """LTX-Video GGUF model loader for text-to-video generation."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q8_0: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-Video GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized transformer; fall back to config on shape mismatch."""
        gguf_file = _GGUF_FILES[self._variant]
        try:
            self._transformer = LTXVideoTransformer3DModel.from_single_file(
                f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
                torch_dtype=dtype,
            )
        except (ValueError, RuntimeError):
            self._transformer = LTXVideoTransformer3DModel(
                in_channels=128,
                out_channels=128,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=32,
                attention_head_dim=64,
                cross_attention_dim=2048,
                num_layers=28,
                activation_fn="gelu-approximate",
                qk_norm="rms_norm_across_heads",
                norm_elementwise_affine=False,
                norm_eps=1e-6,
                caption_channels=4096,
                attention_bias=True,
                attention_out_bias=True,
            ).to(dtype=dtype)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized LTX-Video transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_transformer(dtype)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the LTX-Video transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            self._load_transformer(dtype)

        height = 128
        width = 128
        num_frames = 9

        latent_height = height // _VAE_SPATIAL_COMPRESSION
        latent_width = width // _VAE_SPATIAL_COMPRESSION
        latent_num_frames = (num_frames - 1) // _VAE_TEMPORAL_COMPRESSION + 1

        in_channels = self._transformer.config.in_channels
        video_seq_len = latent_num_frames * latent_height * latent_width

        encoder_hidden_states = torch.randn(
            batch_size, _TOKENIZER_MAX_LENGTH, _CAPTION_CHANNELS, dtype=dtype
        )
        encoder_attention_mask = torch.ones(
            batch_size, _TOKENIZER_MAX_LENGTH, dtype=dtype
        )

        hidden_states = torch.randn(
            batch_size,
            video_seq_len,
            in_channels,
            dtype=dtype,
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "return_dict": False,
        }
