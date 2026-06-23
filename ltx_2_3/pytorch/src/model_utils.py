# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Build helpers and simple-forward wrappers for LTX-2.3 components.

LTX-2.3 (``Lightricks/LTX-2.3``) is a 22B joint audio-video latent-diffusion
foundation model. Unlike LTX-2, it is **not** yet supported by ``diffusers``
("coming soon" per the model card) and ships only as a single bundled
``.safetensors`` checkpoint (the 22B DiT denoiser plus video VAE, audio VAE and
vocoder, with a custom ComfyUI-style ``model.diffusion_model.*`` key layout).
The official ``ltx-core`` package (Lightricks' reference implementation) provides
the module definitions and the single-file loader used here.

Each component is built independently from the same bundled checkpoint via
``ltx_core.loader.SingleGPUModelBuilder`` with a component-specific key filter:

  - DENOISER          -> ltx_core LTXModel (joint AV DiT)            ~18.99B params
  - VIDEO_VAE_DECODER -> ltx_core VideoDecoder                       ~0.41B params
  - AUDIO_VAE_DECODER -> ltx_core AudioDecoder                       ~0.03B params
  - VOCODER           -> ltx_core VocoderWithBWE (mel -> waveform)   ~0.13B params

The text encoder (Gemma-3) is external and is intentionally left out of this
component set; the bundled checkpoint only carries the small
``text_embedding_projection`` connector, not the encoder weights.
"""

from typing import Optional

import torch

from huggingface_hub import hf_hub_download

from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer.model_configurator import (
    LTXModelConfigurator,
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
from ltx_core.model.video_vae.model_configurator import (
    VideoDecoderConfigurator,
    VAE_DECODER_COMFY_KEYS_FILTER,
)
from ltx_core.model.audio_vae.model_configurator import (
    AudioDecoderConfigurator,
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VocoderConfigurator,
    VOCODER_COMFY_KEYS_FILTER,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig

# --- checkpoint ------------------------------------------------------------
LTX23_REPO = "Lightricks/LTX-2.3"
DEV_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"

# Native checkpoint distribution dtype (the dev model is trainable in bf16).
DTYPE = torch.bfloat16

# --- DiT denoiser structural dims (from the checkpoint config) -------------
NUM_HEADS = 32
HEAD_DIM = 128
INNER_DIM = NUM_HEADS * HEAD_DIM            # 4096 -> video context / hidden dim
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = NUM_HEADS * AUDIO_HEAD_DIM  # 2048 -> audio context / hidden dim
IN_CHANNELS = 128                           # video latent channels
AUDIO_IN_CHANNELS = 128                     # audio latent channels (into the DiT)

# Small-but-representative token counts for a single-forward CPU/HW smoke test.
# (Real generation uses far more tokens, derived from the target resolution and
#  frame count; these keep the validation forward cheap.)
VIDEO_TOKENS = 32
AUDIO_TOKENS = 16
CONTEXT_TOKENS = 64

# --- VAE / audio latent layouts (from the checkpoint config) ---------------
VAE_LATENT_CHANNELS = 128                   # video VAE latent channels
AUDIO_VAE_LATENT_CHANNELS = 8               # audio VAE z_channels
AUDIO_VAE_LATENT_MEL = 16                   # mel_bins(64) / 2**(len(ch_mult)-1)
VOCODER_MEL_BINS = 64
VOCODER_STEREO = 2


def _checkpoint_path() -> str:
    """Download (and cache) the bundled LTX-2.3 dev checkpoint, return local path."""
    return hf_hub_download(LTX23_REPO, DEV_CHECKPOINT)


def _build(configurator, sd_ops, dtype: torch.dtype) -> torch.nn.Module:
    """Build a single component from the bundled checkpoint on CPU."""
    builder = SingleGPUModelBuilder(
        model_class_configurator=configurator,
        model_path=_checkpoint_path(),
        model_sd_ops=sd_ops,
    )
    model = builder.build(device=torch.device("cpu"), dtype=dtype)
    return model.eval()


def load_denoiser(dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """Build the joint audio-video DiT denoiser (ltx_core LTXModel)."""
    return _build(LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, dtype)


def load_video_vae_decoder(dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """Build the causal video VAE decoder (latents -> RGB frames)."""
    return _build(VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER, dtype)


def load_audio_vae_decoder(dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """Build the audio VAE decoder (latents -> mel spectrogram)."""
    return _build(AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, dtype)


def load_vocoder(dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """Build the vocoder + bandwidth-extension module (mel -> waveform)."""
    return _build(VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, dtype)


class LTX23DenoiserWrapper(torch.nn.Module):
    """Adapt ``LTXModel`` to a plain-tensor forward for the test harness.

    ``LTXModel.forward`` consumes ``Modality`` dataclasses and a
    ``BatchedPerturbationConfig``; this wrapper rebuilds those from plain
    tensors so the loader can feed a flat dict of tensors and return the raw
    velocity tensors ``(video_velocity, audio_velocity)``.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        video_latent: torch.Tensor,
        video_timesteps: torch.Tensor,
        video_positions: torch.Tensor,
        video_context: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_latent: torch.Tensor,
        audio_timesteps: torch.Tensor,
        audio_positions: torch.Tensor,
        audio_context: torch.Tensor,
        audio_sigma: torch.Tensor,
    ):
        video = Modality(
            latent=video_latent,
            sigma=video_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=torch.ones(
                video_context.shape[0], video_context.shape[1],
                dtype=torch.long, device=video_context.device,
            ),
        )
        audio = Modality(
            latent=audio_latent,
            sigma=audio_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=torch.ones(
                audio_context.shape[0], audio_context.shape[1],
                dtype=torch.long, device=audio_context.device,
            ),
        )
        perturbations = BatchedPerturbationConfig.empty(video_latent.shape[0])
        return self.model(video, audio, perturbations)


def _positions(n_dims: int, tokens: int, hi: int, dtype: torch.dtype) -> torch.Tensor:
    """Build a (B=1, n_dims, tokens, 2) start/end middle-indices RoPE grid."""
    grid = torch.stack(
        [torch.randint(0, hi, (1, tokens)) for _ in range(n_dims)], dim=1
    )
    return grid.unsqueeze(-1).repeat(1, 1, 1, 2).to(dtype)


def denoiser_inputs(dtype: torch.dtype = DTYPE) -> dict:
    """Synthetic single-batch inputs for the DiT denoiser wrapper."""
    return {
        "video_latent": torch.randn(1, VIDEO_TOKENS, IN_CHANNELS, dtype=dtype),
        "video_timesteps": torch.rand(1, VIDEO_TOKENS, dtype=dtype),
        "video_positions": _positions(3, VIDEO_TOKENS, 16, dtype),
        "video_context": torch.randn(1, CONTEXT_TOKENS, INNER_DIM, dtype=dtype),
        "video_sigma": torch.rand(1, dtype=dtype),
        "audio_latent": torch.randn(1, AUDIO_TOKENS, AUDIO_IN_CHANNELS, dtype=dtype),
        "audio_timesteps": torch.rand(1, AUDIO_TOKENS, dtype=dtype),
        "audio_positions": _positions(1, AUDIO_TOKENS, 5, dtype),
        "audio_context": torch.randn(1, CONTEXT_TOKENS, AUDIO_INNER_DIM, dtype=dtype),
        "audio_sigma": torch.rand(1, dtype=dtype),
    }


def video_vae_decoder_inputs(dtype: torch.dtype = DTYPE) -> dict:
    """Latent input for the video VAE decoder: (B, 128, F', H', W')."""
    return {"sample": torch.randn(1, VAE_LATENT_CHANNELS, 2, 16, 16, dtype=dtype)}


def audio_vae_decoder_inputs(dtype: torch.dtype = DTYPE) -> dict:
    """Latent input for the audio VAE decoder: (B, 8, frames, 16)."""
    return {
        "sample": torch.randn(
            1, AUDIO_VAE_LATENT_CHANNELS, 4, AUDIO_VAE_LATENT_MEL, dtype=dtype
        )
    }


def vocoder_inputs(dtype: torch.dtype = DTYPE) -> dict:
    """Stereo mel spectrogram input for the vocoder: (B, 2, T, 64)."""
    return {
        "mel_spec": torch.randn(1, VOCODER_STEREO, 32, VOCODER_MEL_BINS, dtype=dtype)
    }
