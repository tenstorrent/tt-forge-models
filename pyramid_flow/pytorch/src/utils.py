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
# Pyramid Flow SD3 / MMDiT variant
# ============================================================================
#
# Architectural constants matching `rain1011/pyramid-flow-sd3`
# (PyramidDiffusionMMDiT, identical for the 384p and 768p checkpoints, see
#  https://huggingface.co/rain1011/pyramid-flow-sd3/blob/main/diffusion_transformer_768p/config.json).
#
# The non-config kwargs (add_temp_pos_embed / temp_pos_embed_type / use_t5_mask)
# are the ones the upstream pipeline injects at `from_pretrained` time
# (see Pyramid-Flow `pyramid_dit_for_video_gen_pipeline.build_pyramid_dit`).
# use_flash_attn is forced False: flash-attn is unavailable here and the
# SDPA fallback path (VarlenSelfAttentionWithT5Mask) is the device target.

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
    use_gradient_checkpointing=False,
    use_flash_attn=False,
    use_temporal_causal=True,
    use_t5_mask=True,
    add_temp_pos_embed=True,
    interp_condition_pos=True,
)

# Smoke-test latent shape (single pyramid stage, single clip).
_MMDIT_SMOKE_TEMP = 1
_MMDIT_SMOKE_HEIGHT = 32  # latent H; patched to 16 internally (patch_size=2)
_MMDIT_SMOKE_WIDTH = 32
_MMDIT_SMOKE_TEXT_SEQ_LEN = 16
_MMDIT_SMOKE_BATCH = 1


def load_mmdit_transformer(dtype: torch.dtype) -> PyramidDiffusionMMDiT:
    """
    Instantiate the PyramidDiffusionMMDiT (SD3) DiT with random weights.

    Pyramid Flow has no diffusers integration; the model code is vendored
    locally. Weights are random — sufficient for compilation / op-coverage
    error analysis, not for accuracy.
    """
    model = PyramidDiffusionMMDiT(**MMDIT_CONFIG)
    model = model.to(dtype=dtype).eval()
    return model


def load_mmdit_transformer_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """
    Build synthetic inputs for a single-stage Pyramid Flow MMDiT forward pass.

    Returns a dict matching `PyramidDiffusionMMDiT.forward`. The `sample` field
    is the upstream pyramid-stage list `[stage_0, stage_1, ...]`, each stage a
    `[B, C_latent, T, H, W]` latent tensor. We use a single stage for the smoke
    variant.
    """
    cfg = MMDIT_CONFIG
    batch_size = _MMDIT_SMOKE_BATCH
    seq_len = _MMDIT_SMOKE_TEXT_SEQ_LEN

    sample = [
        torch.randn(
            batch_size,
            cfg["in_channels"],
            _MMDIT_SMOKE_TEMP,
            _MMDIT_SMOKE_HEIGHT,
            _MMDIT_SMOKE_WIDTH,
            dtype=dtype,
        )
    ]
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, cfg["joint_attention_dim"], dtype=dtype
    )
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pooled_projections = torch.randn(
        batch_size, cfg["pooled_projection_dim"], dtype=dtype
    )
    timestep_ratio = torch.tensor([500.0] * batch_size, dtype=dtype)

    return {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "pooled_projections": pooled_projections,
        "timestep_ratio": timestep_ratio,
    }
