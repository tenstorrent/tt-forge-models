# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 transformer loading and input generation utilities.

Architecture constants are derived from the LTX-Video VAE family.

The loader bypasses the DistilledPipeline (which requires CUDA-capable torchaudio
and local file paths) and loads the transformer directly from the safetensors
checkpoint using ltx_core.loader.SingleGPUModelBuilder.
"""

import sys
import types
from typing import Dict, Tuple

import torch

# VAE compression ratios (LTX-Video / LTX-2 family)
SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 8

# Number of latent channels output by the VAE encoder
LATENT_CHANNELS = 128

# Gemma 3 text encoder (used by LTX-2.3).
# TEXT_HIDDEN_DIM matches the 4B variant; update if a larger encoder is used.
TEXT_HIDDEN_DIM = 3072
TEXT_SEQ_LEN = 256

# Default test resolution — constraints:
#   height % 32 == 0, width % 32 == 0
#   (num_frames - 1) % 8 == 0  →  9, 17, 25, 33, ...
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_NUM_FRAMES = 9
DEFAULT_BATCH_SIZE = 1


def compute_latent_dims(
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> Tuple[int, int, int, int]:
    """Return (lat_h, lat_w, lat_t, seq_len) for the given video resolution."""
    assert (
        height % SPATIAL_COMPRESSION == 0
    ), f"height must be divisible by {SPATIAL_COMPRESSION}, got {height}"
    assert (
        width % SPATIAL_COMPRESSION == 0
    ), f"width must be divisible by {SPATIAL_COMPRESSION}, got {width}"
    assert (
        num_frames - 1
    ) % TEMPORAL_COMPRESSION == 0, (
        f"num_frames must satisfy 8k+1 (e.g. 9, 17, 25), got {num_frames}"
    )
    lat_h = height // SPATIAL_COMPRESSION
    lat_w = width // SPATIAL_COMPRESSION
    lat_t = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    return lat_h, lat_w, lat_t, lat_t * lat_h * lat_w


def _stub_audio_modules() -> None:
    """Stub ltx_core audio submodules that transitively import torchaudio.

    torchaudio is installed but its native extension requires libcudart.so.13
    which is unavailable on this host. We only need the video transformer, so
    audio code paths are never exercised.
    """
    if "ltx_core.model.audio_vae.ops" not in sys.modules:
        stub = types.ModuleType("ltx_core.model.audio_vae.ops")
        # Sentinel classes used in type annotations within audio_vae.audio_vae.
        stub.AudioProcessor = type("AudioProcessor", (), {})
        stub.PerChannelStatistics = type("PerChannelStatistics", (), {})
        sys.modules["ltx_core.model.audio_vae.ops"] = stub


def load_transformer_direct(
    hf_repo: str,
    checkpoint_filename: str,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Download checkpoint from HuggingFace and build the LTX-2.3 transformer.

    Uses ltx_core.loader.SingleGPUModelBuilder with LTXModelConfigurator to
    instantiate the DiT directly from the safetensors file, without requiring
    the full DistilledPipeline.
    """
    _stub_audio_modules()

    from huggingface_hub import hf_hub_download

    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=checkpoint_filename)

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=checkpoint_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    transformer = builder.build(device=device, dtype=dtype)
    return transformer


def load_transformer_inputs(
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = DEFAULT_BATCH_SIZE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_frames: int = DEFAULT_NUM_FRAMES,
    guidance_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic transformer inputs for LTX-2.3.

    hidden_states shape follows the LTX-Video 0.9.x spatial convention:
    [B, C, T, H, W] — the transformer handles patchification internally.

    When guidance_scale > 1 (CFG active), the batch is doubled to hold
    [unconditional, conditional] pairs, matching the pipeline's behaviour.
    """
    lat_h, lat_w, lat_t, _ = compute_latent_dims(height, width, num_frames)
    b = batch_size * 2 if guidance_scale > 1.0 else batch_size

    return {
        "hidden_states": torch.randn(
            b, LATENT_CHANNELS, lat_t, lat_h, lat_w, dtype=dtype
        ),
        "encoder_hidden_states": torch.randn(
            b, TEXT_SEQ_LEN, TEXT_HIDDEN_DIM, dtype=dtype
        ),
        "encoder_attention_mask": torch.ones(b, TEXT_SEQ_LEN, dtype=torch.bool),
        "timestep": torch.full((b,), 0.5, dtype=dtype),
        "num_frames": lat_t,
        "height": lat_h,
        "width": lat_w,
    }
