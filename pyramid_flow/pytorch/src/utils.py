# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Pyramid Flow model loading."""

from typing import Any, Dict

import torch

from .flux_modules import PyramidFluxTransformer
from .mmdit_modules import PyramidDiffusionMMDiT

# ============================================================================
# Architectural constants matching `rain1011/pyramid-flow-miniflux`
# (Pyramid Flow miniFLUX-768p DiT, see
#  https://huggingface.co/rain1011/pyramid-flow-miniflux/blob/main/diffusion_transformer_768p/config.json)
# ============================================================================

DIT_CONFIG = dict(
    patch_size=1,
    in_channels=64,
    num_layers=19,
    num_single_layers=38,
    attention_head_dim=128,
    num_attention_heads=24,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    axes_dims_rope=[16, 56, 56],
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
    Instantiate the PyramidFluxTransformer DiT with random weights.

    Pyramid Flow has no diffusers integration; we vendor the model code
    locally and instantiate from scratch. Weights are random — sufficient for
    compilation / op-coverage error analysis, not for accuracy.
    """
    model = PyramidFluxTransformer(**DIT_CONFIG)
    model = model.to(dtype=dtype).eval()
    return model


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


# ============================================================================
# SD3 MMDiT variant (`rain1011/pyramid-flow-sd3`)
# ============================================================================
#
# Architectural constants for the Pyramid Flow SD3 MMDiT denoiser. The fields
# in `diffusion_transformer_768p/config.json` are augmented with the extra
# constructor kwargs the upstream pipeline passes when it builds the SD3 DiT
# (see `pyramid_dit/pyramid_dit_for_video_gen_pipeline.py`):
# `add_temp_pos_embed=True`, `temp_pos_embed_type='rope'`, `use_t5_mask=True`,
# `interp_condition_pos=True`, `use_temporal_causal=True`, `use_flash_attn=False`.
# The 384p and 768p checkpoints share an identical DiT config; they differ only
# in the generation resolution, so a single config serves both.

MMDIT_CONFIG = dict(
    sample_size=128,
    patch_size=2,
    in_channels=16,
    num_layers=24,
    attention_head_dim=64,
    num_attention_heads=24,
    caption_projection_dim=1536,
    pooled_projection_dim=2048,
    pos_embed_max_size=192,
    max_num_frames=200,
    qk_norm="rms_norm",
    pos_embed_type="sincos",
    temp_pos_embed_type="rope",
    joint_attention_dim=4096,
    use_flash_attn=False,
    use_temporal_causal=True,
    use_t5_mask=True,
    add_temp_pos_embed=True,
    interp_condition_pos=True,
    use_gradient_checkpointing=False,
)

# Smoke-test latent shape for the SD3 MMDiT (single pyramid stage, single clip).
# Latent channels are the VAE's 16 channels; spatial dims must be a multiple of
# patch_size (2). T5 text-token sequence length feeds joint_attention_dim.
_MMDIT_SMOKE_TEMP = 1
_MMDIT_SMOKE_HEIGHT = 32
_MMDIT_SMOKE_WIDTH = 32
_MMDIT_SMOKE_TEXT_SEQ_LEN = 16
_MMDIT_SMOKE_BATCH = 1


def load_mmdit_transformer(dtype: torch.dtype) -> PyramidDiffusionMMDiT:
    """
    Instantiate the PyramidDiffusionMMDiT (SD3) denoiser with random weights.

    Pyramid Flow has no diffusers integration; we vendor the model code locally
    and instantiate from scratch. Weights are random — sufficient for
    compilation / op-coverage error analysis, not for accuracy.
    """
    model = PyramidDiffusionMMDiT(**MMDIT_CONFIG)
    model = model.to(dtype=dtype).eval()
    return model


def load_mmdit_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """
    Build synthetic inputs for a single-stage Pyramid Flow SD3 MMDiT forward.

    Returns a dict matching `PyramidDiffusionMMDiT.forward`. The `sample` field
    is a list-of-lists per the upstream pyramid-stage structure:
    `[stage_0_clips, stage_1_clips, ...]` where each clip is a
    `[B, C_latent, T, H, W]` tensor. We use a single stage with a single clip
    for the smoke variant. `encoder_hidden_states` carries the T5 sequence
    (joint_attention_dim), `pooled_projections` the concatenated CLIP-L/CLIP-G
    pooled embeddings (pooled_projection_dim).
    """
    cfg = MMDIT_CONFIG
    batch_size = _MMDIT_SMOKE_BATCH
    seq_len = _MMDIT_SMOKE_TEXT_SEQ_LEN

    sample = [
        [
            torch.randn(
                batch_size,
                cfg["in_channels"],
                _MMDIT_SMOKE_TEMP,
                _MMDIT_SMOKE_HEIGHT,
                _MMDIT_SMOKE_WIDTH,
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
