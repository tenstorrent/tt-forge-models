# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Pyramid Flow model loading."""

from typing import Any, Dict

import torch

from .flux_modules import PyramidFluxTransformer


# HuggingFace repo carrying the real Pyramid Flow miniFLUX weights (ungated).
_HF_REPO = "rain1011/pyramid-flow-miniflux"
# Variant -> the DiT subfolder inside the repo (each has config.json +
# diffusion_pytorch_model.safetensors). 768p is the default bringup target.
_DIT_SUBFOLDER = "diffusion_transformer_768p"


# ============================================================================
# Architectural constants matching `rain1011/pyramid-flow-miniflux`
# (Pyramid Flow miniFLUX-768p DiT, see
#  https://huggingface.co/rain1011/pyramid-flow-miniflux/blob/main/diffusion_transformer_768p/config.json)
# ============================================================================

DIT_CONFIG = dict(
    patch_size=1,
    in_channels=64,
    num_layers=8,
    num_single_layers=16,
    attention_head_dim=64,
    num_attention_heads=30,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    axes_dims_rope=[16, 24, 24],
    use_flash_attn=False,
    use_temporal_causal=True,
    interp_condition_pos=True,
    use_gradient_checkpointing=False,
)


# Internal-patch dimension is hard-coded to 2 in PyramidFluxTransformer; latent
# channels visible to the user are `in_channels // (patch * patch)`.
_INTERNAL_PATCH = 2

# Smoke-test latent shape (single pyramid stage).
_SMOKE_TEMP = 1
_SMOKE_HEIGHT = 16
_SMOKE_WIDTH = 16
_SMOKE_TEXT_SEQ_LEN = 16
_SMOKE_BATCH = 1


# ============================================================================
# Model loading
# ============================================================================


def load_transformer(dtype: torch.dtype) -> PyramidFluxTransformer:
    """
    Load the real Pyramid Flow miniFLUX DiT with pretrained weights.

    The weights come from the public (ungated) `rain1011/pyramid-flow-miniflux`
    HuggingFace repo via `from_pretrained`. If the download fails (e.g. no
    network), we fall back to a randomly-initialised model with the corrected
    miniFLUX config so compilation / op-coverage analysis can still run.
    """
    try:
        model = PyramidFluxTransformer.from_pretrained(
            _HF_REPO,
            subfolder=_DIT_SUBFOLDER,
            torch_dtype=dtype,
        )
    except Exception:
        model = PyramidFluxTransformer(**DIT_CONFIG).to(dtype=dtype)
    return model.eval()


# ============================================================================
# Input loading
# ============================================================================


def load_transformer_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """
    Build synthetic inputs for a single-stage Pyramid Flow DiT forward pass.

    Returns a dict matching `PyramidFluxTransformer.forward` signature.
    The `sample` field is a list-of-lists per the upstream pyramid-stage
    structure: `[stage_0_clips, stage_1_clips, ...]` where each clip is a
    `[B, C_latent, T, H, W]` tensor. We use a single stage with a single
    clip for the smoke variant.
    """
    cfg = DIT_CONFIG
    batch_size = _SMOKE_BATCH
    latent_channels = cfg["in_channels"] // (_INTERNAL_PATCH * _INTERNAL_PATCH)
    seq_len = _SMOKE_TEXT_SEQ_LEN

    sample = [
        [
            torch.randn(
                batch_size,
                latent_channels,
                _SMOKE_TEMP,
                _SMOKE_HEIGHT,
                _SMOKE_WIDTH,
                dtype=dtype,
            )
        ]
    ]
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, cfg["joint_attention_dim"], dtype=dtype
    )
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pooled_projections = torch.randn(
        batch_size, cfg["pooled_projection_dim"], dtype=dtype
    )
    timestep_ratio = torch.tensor([500.0], dtype=dtype)

    return {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "pooled_projections": pooled_projections,
        "timestep_ratio": timestep_ratio,
    }
