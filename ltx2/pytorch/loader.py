# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 model loader for text-to-video generation.

LTX-2 is a 19B dual-stream DiT (video + audio) by Lightricks.
Repository: https://github.com/Lightricks/LTX-2
Weights:    https://huggingface.co/Lightricks/LTX-2

Variants
--------
FAST     — distilled checkpoint, 8 denoising steps, guidance_scale=1.0
STANDARD — full checkpoint, 40 denoising steps, guidance_scale=4.0

Both share the LTX2VideoTransformer3DModel architecture (19B dual-stream DiT).
Only checkpoint weights and inference step count differ.
"""
import torch
from diffusers import LTX2VideoTransformer3DModel
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

# VAE compression ratios — LTX-2 family (confirmed from LTX2Pipeline defaults)
_VAE_SPATIAL_COMPRESSION = 32
_VAE_TEMPORAL_COMPRESSION = 8
_AUDIO_VAE_TEMPORAL_COMPRESSION = 4
_AUDIO_VAE_MEL_COMPRESSION = 4
_AUDIO_SAMPLING_RATE = 16000
_AUDIO_HOP_LENGTH = 160
_AUDIO_MEL_BINS = 64

# Minimum valid dimensions for TT hardware bringup:
#   (num_frames - 1) % temporal_compression == 0  →  9, 17, 25, ...
#   height, width % spatial_compression == 0       →  32, 64, 96, ...
DEFAULT_NUM_FRAMES = 9
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_FRAME_RATE = 24.0
DEFAULT_SEQ_LEN = 128

_HF_REPO = "Lightricks/LTX-2"


class ModelVariant(StrEnum):
    FAST = "Fast"  # distilled, 8-step
    STANDARD = "Standard"


class ModelLoader(ForgeModel):
    """LTX-2 transformer loader (19B dual-stream DiT, video + audio)."""

    _VARIANTS = {
        ModelVariant.FAST: ModelConfig(pretrained_model_name=_HF_REPO),
        ModelVariant.STANDARD: ModelConfig(pretrained_model_name=_HF_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.transformer = LTX2VideoTransformer3DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        return self.transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LTX-2 DiT transformer (compilation target).

        Returns:
            LTX2VideoTransformer3DModel
        """
        if self.transformer is None:
            self._load_transformer(dtype_override=dtype_override)
        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build synthetic inputs for one LTX-2 transformer forward pass.

        Uses minimal spatial/temporal dimensions to fit on device.
        Text embeddings are synthetic tensors matching the connector output shape.
        video_coords/audio_coords are omitted — the transformer computes them
        internally from num_frames/height/width.

        Returns:
            dict: kwargs for LTX2VideoTransformer3DModel.forward()
        """
        if self.transformer is None:
            self._load_transformer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # ── Video latent dims ──────────────────────────────────────────────
        latent_num_frames = (
            DEFAULT_NUM_FRAMES - 1
        ) // _VAE_TEMPORAL_COMPRESSION + 1  # 2
        latent_height = DEFAULT_HEIGHT // _VAE_SPATIAL_COMPRESSION  # 1
        latent_width = DEFAULT_WIDTH // _VAE_SPATIAL_COMPRESSION  # 1
        video_tokens = latent_num_frames * latent_height * latent_width  # 2

        video_in_channels = self.transformer.config.in_channels  # 128
        # Packed latents: [B, video_tokens, in_channels]
        hidden_states = torch.randn(
            batch_size, video_tokens, video_in_channels, dtype=dtype
        )

        # ── Audio latent dims ──────────────────────────────────────────────
        duration_s = DEFAULT_NUM_FRAMES / DEFAULT_FRAME_RATE
        audio_latents_per_second = (
            _AUDIO_SAMPLING_RATE
            / _AUDIO_HOP_LENGTH
            / float(_AUDIO_VAE_TEMPORAL_COMPRESSION)
        )
        audio_num_frames = max(1, round(duration_s * audio_latents_per_second))  # ~9
        latent_mel_bins = _AUDIO_MEL_BINS // _AUDIO_VAE_MEL_COMPRESSION  # 16
        audio_tokens = audio_num_frames * latent_mel_bins

        audio_in_channels = self.transformer.config.audio_in_channels  # 128
        # Packed audio latents: [B, audio_tokens, audio_in_channels]
        audio_hidden_states = torch.randn(
            batch_size, audio_tokens, audio_in_channels, dtype=dtype
        )

        # ── Text embeddings ────────────────────────────────────────────────
        # Shape matches connector output: [B, seq_len, caption_channels]
        # The transformer's caption_projection reduces this to inner_dim.
        caption_channels = self.transformer.config.caption_channels  # 3840
        encoder_hidden_states = torch.randn(
            batch_size, DEFAULT_SEQ_LEN, caption_channels, dtype=dtype
        )
        audio_encoder_hidden_states = torch.randn(
            batch_size, DEFAULT_SEQ_LEN, caption_channels, dtype=dtype
        )
        # Binary mask: 1 = attend, 0 = ignore.
        # The transformer converts this to additive form: (1-mask)*-10000.
        encoder_attention_mask = torch.ones(batch_size, DEFAULT_SEQ_LEN, dtype=dtype)

        # ── Timestep ──────────────────────────────────────────────────────
        # Shape [B, video_tokens]; scaled by timestep_scale_multiplier=1000.
        # flatten() inside the transformer handles both [B] and [B, T] shapes.
        timestep = torch.full((batch_size, video_tokens), 1000.0, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "audio_encoder_attention_mask": encoder_attention_mask,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": DEFAULT_FRAME_RATE,
            "audio_num_frames": audio_num_frames,
            # video_coords / audio_coords omitted: transformer computes them internally
            "return_dict": False,
        }

    def unpack_forward_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
