# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for loading the Pyramid Flow SD3 (MMDiT) denoiser.

The SD3 variant of Pyramid Flow (``rain1011/pyramid-flow-sd3``) uses a
``PyramidDiffusionMMDiT`` denoiser — an SD3-style MMDiT (joint text/image
transformer) rather than the miniFLUX DiT of the other variant. The model code
is vendored in ``mmdit_modules/`` (verbatim from upstream, with the
``trainer_misc`` sequence-parallel imports replaced by a local stub and the
``IPython`` debug import dropped). This module loads the real pretrained weights
so the on-device PCC comparison is meaningful.
"""

import os
from typing import Any, Dict

import torch
from huggingface_hub import snapshot_download

from .mmdit_modules import PyramidDiffusionMMDiT

# ============================================================================
# Native 384p latent geometry (rain1011/pyramid-flow-sd3, 384p DiT)
#
# Pyramid Flow 384p generates 384x640 video; the causal video VAE downsamples
# spatially by 8, giving a 48x80 latent. The autoregressive temporal scheme
# denoises one temporal unit (frame_per_unit=1 -> T=1 latent frame) per DiT
# call, so a single-stage, single-clip [B, 16, 1, 48, 80] latent is one real
# per-step shape at native spatial resolution (not a downscale).
# ============================================================================

_LATENT_CHANNELS = 16  # mmdit in_channels (latent channels directly)
_LATENT_TEMP = 1  # one temporal unit per denoise step
_LATENT_HEIGHT = 48  # 384 / 8
_LATENT_WIDTH = 80  # 640 / 8
_TEXT_SEQ_LEN = 128  # SD3 T5 token budget used by Pyramid Flow
_JOINT_ATTENTION_DIM = 4096  # T5-XXL d_model
_POOLED_PROJECTION_DIM = 2048  # CLIP-L (768) + CLIP-G (1280) pooled
_BATCH = 1

# Extra constructor kwargs the upstream pipeline passes to
# PyramidDiffusionMMDiT.from_pretrained on top of config.json (see
# pyramid_dit/pyramid_dit_for_video_gen_pipeline.py::build_pyramid_dit). These
# select the temporal-rope path the 384p model actually runs with.
_DIT_BUILD_KWARGS = dict(
    use_flash_attn=False,
    use_t5_mask=True,
    add_temp_pos_embed=True,
    temp_pos_embed_type="rope",
    use_temporal_causal=True,
    interp_condition_pos=True,
)

_HF_REPO = "rain1011/pyramid-flow-sd3"
_DIT_SUBFOLDER = "diffusion_transformer_384p"


def _resolve_dit_dir() -> str:
    """Download (or reuse cached) the 384p DiT component and return its dir."""
    root = snapshot_download(
        _HF_REPO, allow_patterns=[f"{_DIT_SUBFOLDER}/*"]
    )
    return os.path.join(root, _DIT_SUBFOLDER)


def load_transformer(dtype: torch.dtype) -> PyramidDiffusionMMDiT:
    """Load the PyramidDiffusionMMDiT (SD3 384p) denoiser with real weights."""
    dit_dir = _resolve_dit_dir()
    model = PyramidDiffusionMMDiT.from_pretrained(
        dit_dir,
        torch_dtype=dtype,
        **_DIT_BUILD_KWARGS,
    )
    model = model.to(dtype=dtype).eval()
    return model


def load_transformer_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """Build a single-stage, single-clip MMDiT forward input at native 384p.

    Matches ``PyramidDiffusionMMDiT.forward``: ``sample`` is the pyramid
    list-of-stages structure (one stage, one clip here), each clip a
    ``[B, C_latent, T, H_lat, W_lat]`` tensor.
    """
    sample = [
        [
            torch.randn(
                _BATCH,
                _LATENT_CHANNELS,
                _LATENT_TEMP,
                _LATENT_HEIGHT,
                _LATENT_WIDTH,
                dtype=dtype,
            )
        ]
    ]
    encoder_hidden_states = torch.randn(
        _BATCH, _TEXT_SEQ_LEN, _JOINT_ATTENTION_DIM, dtype=dtype
    )
    encoder_attention_mask = torch.ones(_BATCH, _TEXT_SEQ_LEN, dtype=torch.long)
    pooled_projections = torch.randn(
        _BATCH, _POOLED_PROJECTION_DIM, dtype=dtype
    )
    timestep_ratio = torch.tensor([500.0], dtype=dtype)

    return {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "pooled_projections": pooled_projections,
        "timestep_ratio": timestep_ratio,
    }
