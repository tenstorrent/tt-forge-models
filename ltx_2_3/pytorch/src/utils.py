# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for the LTX-2.3 (Lightricks) audio-video generation model.

LTX-2.3 is a 22B DiT-based *joint audio-video* foundation model. Unlike the
diffusers-based video models (Mochi, CogVideoX), LTX-2.3 is NOT yet supported by
the `diffusers` library ("coming soon" per the model card) and ships as a single
raw LTX-native `.safetensors` checkpoint (no diffusers component layout).

All non-text components — the DiT transformer (denoiser), the video VAE
(encoder/decoder), the audio VAE (encoder/decoder) and the HiFi-GAN vocoder — are
packed into the one `ltx-2.3-22b-dev.safetensors` (46 GB) checkpoint and are
extracted by key-prefix filters. The per-component model config is read from the
safetensors metadata header. The Gemma-3-12B text encoder lives in a *separate*
(gated, local) `gemma_root` directory and is therefore optional here.

Components are built with `ltx_core`'s `SingleGPUModelBuilder` (the same machinery
the official `ltx_pipelines.ModelLedger` uses), so we depend only on `ltx-core`,
not on the heavier `ltx-pipelines` package.

Reference: https://huggingface.co/Lightricks/LTX-2.3 and the LTX-2 codebase
https://github.com/Lightricks/LTX-2 (packages `ltx-core`, `ltx-pipelines`).
"""

from typing import Optional

import torch

# Default repo / checkpoint. All in-checkpoint components share this file.
DEFAULT_REPO = "Lightricks/LTX-2.3"
DEFAULT_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"

# Video VAE geometry (from the checkpoint config):
#   latent_channels = 128
#   spatial compression = 32x  (H = 32 * H_latent, W = 32 * W_latent)
#   temporal compression = 8x  (F = 8 * (F_latent - 1) + 1)
VAE_LATENT_CHANNELS = 128
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8

# Native single-stage generation defaults (constants.py of ltx-pipelines):
#   width=768, height=512, num_frames=121, num_inference_steps=40, fps=24
NATIVE_WIDTH = 768
NATIVE_HEIGHT = 512
NATIVE_FRAMES = 121
NATIVE_LATENT_W = NATIVE_WIDTH // VAE_SPATIAL_COMPRESSION  # 24
NATIVE_LATENT_H = NATIVE_HEIGHT // VAE_SPATIAL_COMPRESSION  # 16
NATIVE_LATENT_F = 1 + (NATIVE_FRAMES - 1) // VAE_TEMPORAL_COMPRESSION  # 16


# ---------------------------------------------------------------------------
# Checkpoint download + component builders
# ---------------------------------------------------------------------------


def download_checkpoint(repo: str = DEFAULT_REPO, filename: str = DEFAULT_CHECKPOINT) -> str:
    """Download (or hit the cache for) the LTX-2.3 checkpoint and return its path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def _build(checkpoint_path, configurator, sd_ops, dtype, device):
    """Build a single component from the checkpoint via ltx_core's builder.

    The config is read from the safetensors metadata header; `sd_ops` selects and
    renames the component's weight keys out of the combined checkpoint.
    """
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

    builder = SingleGPUModelBuilder(
        model_path=checkpoint_path,
        model_class_configurator=configurator,
        model_sd_ops=sd_ops,
    )
    return builder.build(device=device, dtype=dtype).eval()


def load_video_decoder(checkpoint_path: str, dtype: torch.dtype, device=None):
    """Build the video VAE decoder: latents (B,128,F',H',W') -> video (B,3,F,H,W)."""
    from ltx_core.model.video_vae import (
        VAE_DECODER_COMFY_KEYS_FILTER,
        VideoDecoderConfigurator,
    )

    device = device or torch.device("cpu")
    model = _build(
        checkpoint_path, VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER, dtype, device
    )
    return model


def load_video_encoder(checkpoint_path: str, dtype: torch.dtype, device=None):
    """Build the video VAE encoder: video (B,3,F,H,W) -> latents (B,128,F',H',W')."""
    from ltx_core.model.video_vae import (
        VAE_ENCODER_COMFY_KEYS_FILTER,
        VideoEncoderConfigurator,
    )

    device = device or torch.device("cpu")
    return _build(
        checkpoint_path, VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER, dtype, device
    )


def load_transformer(checkpoint_path: str, dtype: torch.dtype, device=None):
    """Build the 22B DiT denoiser, wrapped in X0Model (x0-prediction).

    NOTE: this is a *joint audio-video* DiT whose forward takes `Modality`
    dataclasses (video, audio) and a `BatchedPerturbationConfig` — it is NOT a
    plain `model(**dict)` forward and is ~44 GB in bf16 (needs multi-chip
    sharding to fit Blackhole). Provided for completeness / future bringup.
    """
    from ltx_core.model.transformer import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
        X0Model,
    )

    device = device or torch.device("cpu")
    base = _build(
        checkpoint_path, LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, dtype, device
    )
    return X0Model(base).to(device).eval()


def load_audio_decoder(checkpoint_path: str, dtype: torch.dtype, device=None):
    """Build the audio VAE decoder."""
    from ltx_core.model.audio_vae import (
        AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        AudioDecoderConfigurator,
    )

    device = device or torch.device("cpu")
    return _build(
        checkpoint_path, AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER, dtype, device
    )


def load_vocoder(checkpoint_path: str, dtype: torch.dtype, device=None):
    """Build the HiFi-GAN vocoder (mel -> waveform)."""
    from ltx_core.model.audio_vae import VOCODER_COMFY_KEYS_FILTER, VocoderConfigurator

    device = device or torch.device("cpu")
    return _build(
        checkpoint_path, VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER, dtype, device
    )


def disable_vae_noise(model) -> None:
    """Make the video VAE decoder deterministic for CPU-vs-TT comparison.

    The decoder injects gaussian noise in two places when `timestep_conditioning`
    is on: a top-level `decode_noise_scale` term and per-resnet-block
    `inject_noise`. Both call `torch.randn`, which would make CPU and device
    outputs differ run-to-run and is awkward on XLA. Zero them out so the forward
    is a pure deterministic function of the latent.
    """
    if hasattr(model, "decode_noise_scale"):
        model.decode_noise_scale = 0.0
    for m in model.modules():
        if hasattr(m, "inject_noise"):
            m.inject_noise = False


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def video_decoder_latent(
    dtype: torch.dtype,
    latent_f: int = NATIVE_LATENT_F,
    latent_h: int = NATIVE_LATENT_H,
    latent_w: int = NATIVE_LATENT_W,
) -> torch.Tensor:
    """A VAE-decoder latent (B,128,F',H',W'). Defaults to native single-stage res."""
    return torch.randn(1, VAE_LATENT_CHANNELS, latent_f, latent_h, latent_w, dtype=dtype)


def video_encoder_pixels(
    dtype: torch.dtype,
    frames: int = NATIVE_FRAMES,
    height: int = NATIVE_HEIGHT,
    width: int = NATIVE_WIDTH,
) -> torch.Tensor:
    """An RGB video clip (B,3,F,H,W) for the VAE encoder. (F-1) must be div by 8."""
    return torch.randn(1, 3, frames, height, width, dtype=dtype)


# Transformer (DiT) cross-attention dim for the video stream (config).
TRANSFORMER_VIDEO_CROSS_DIM = 4096
TRANSFORMER_TEXT_SEQ_LEN = 128


def transformer_video_inputs(
    dtype: torch.dtype,
    latent_f: int = 2,
    latent_h: int = 8,
    latent_w: int = 8,
    seq_len: int = TRANSFORMER_TEXT_SEQ_LEN,
) -> dict:
    """Build a minimal *video-only* DiT forward call: {video, audio, perturbations}.

    The DiT consumes `Modality` dataclasses, not flat tensors. This mirrors what
    `ltx_pipelines` does (patchify the latent, derive the RoPE position grid from
    `get_patch_grid_bounds`, attach text `context`). `audio` is None (video-only).
    Defaults to a small clip so a CPU forward is fast; the 18.99B model needs
    multi-chip sharding to run on device.
    """
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import VideoLatentShape

    b = 1
    patch = VideoLatentPatchifier(patch_size=1)
    latent = torch.randn(b, VAE_LATENT_CHANNELS, latent_f, latent_h, latent_w, dtype=dtype)
    tokens = patch.patchify(latent)  # (B, T, 128)
    num_tokens = tokens.shape[1]
    # RoPE position grid (B, 3, T, 2) of [start, end) patch bounds.
    positions = patch.get_patch_grid_bounds(
        VideoLatentShape(
            batch=b, channels=VAE_LATENT_CHANNELS, frames=latent_f, height=latent_h, width=latent_w
        )
    )
    video = Modality(
        latent=tokens,
        sigma=torch.ones(b, dtype=dtype),
        timesteps=torch.ones(b, num_tokens, dtype=dtype),
        positions=positions,
        context=torch.randn(b, seq_len, TRANSFORMER_VIDEO_CROSS_DIM, dtype=dtype),
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )
    return {
        "video": video,
        "audio": None,
        "perturbations": BatchedPerturbationConfig.empty(b),
    }
