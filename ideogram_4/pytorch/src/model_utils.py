# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ideogram 4 component loaders.

Weights are published as weight-only FP8 (e4m3 + per-row float32 scales) in
``ideogram-ai/ideogram-4-fp8``. For tt-xla bringup we materialize those
linear weights to bfloat16 at load time, then let the compiler lower them to
TT block formats (bfp_bf8 / bfp_bf4) via mixed-precision overrides.
"""

from __future__ import annotations

import json
from typing import Dict

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# NOTE: the ``ideogram4`` package (declared in requirements.txt) is imported
# lazily inside the functions that need it, not at module top-level. Test
# discovery imports this loader before per-model requirements are installed, so
# a top-level ``import ideogram4`` would drop the model from parametrization.
# Mirrors the pi_0 loader convention for git-installed packages.

REPO_ID = "ideogram-ai/ideogram-4-fp8"
DTYPE = torch.bfloat16

# Shapes for 512x512 generation (patch_size=2, ae_scale_factor=8 → patch=16).
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8
PATCH = PATCH_SIZE * AE_SCALE_FACTOR  # 16
IMAGE_H = 512
IMAGE_W = 512
GRID_H = IMAGE_H // PATCH  # 32
GRID_W = IMAGE_W // PATCH  # 32
NUM_IMAGE_TOKENS = GRID_H * GRID_W  # 1024
MAX_TEXT_TOKENS = 256
TOTAL_SEQ_LEN = MAX_TEXT_TOKENS + NUM_IMAGE_TOKENS  # 1280

IN_CHANNELS = 128
LLM_FEATURES_DIM = 4096 * 13  # Qwen3-VL hidden × tapped layers

FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"

# JSON caption used for CPU golden / e2e smoke (matches bringup run).
DEFAULT_JSON_CAPTION = json.dumps(
    {
        "high_level_description": (
            "A ginger cat wearing a tiny wizard hat reading a spellbook."
        ),
        "style_description": {
            "aesthetics": "whimsical, warm, cozy",
            "lighting": "soft indoor light",
            "photo": "eye-level, shallow depth of field",
            "medium": "digital illustration",
            "color_palette": ["#F4A460", "#8B4513", "#FFFFFF", "#4B0082", "#FFD700"],
        },
        "compositional_deconstruction": {
            "background": (
                "A cozy library nook with wooden shelves and warm lamplight."
            ),
            "elements": [
                {
                    "type": "obj",
                    "bbox": [250, 250, 750, 850],
                    "desc": (
                        "A fluffy ginger cat with a tiny purple wizard hat, "
                        "paws on an open spellbook."
                    ),
                }
            ],
        },
    },
    separators=(",", ":"),
    ensure_ascii=False,
)


def _load_sharded_state_dict(
    repo_id: str, index_filename: str
) -> Dict[str, torch.Tensor]:
    index_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_dir = index_filename.rsplit("/", 1)[0] if "/" in index_filename else ""
    state_dict: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{shard_dir}/{shard}" if shard_dir else shard,
        )
        state_dict.update(load_file(shard_path))
    return state_dict


def materialize_fp8_state_dict_to_bf16(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert weight-only FP8 linear checkpoints into plain bf16 tensors."""
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(FP8_SCALE_SUFFIX):
            continue
        if tensor.dtype == FP8_WEIGHT_DTYPE:
            scale_key = key + "_scale"
            scale = state_dict[scale_key]
            dequant = tensor.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)
            out[key] = dequant.to(torch.bfloat16)
            continue
        if tensor.is_floating_point():
            out[key] = tensor.to(torch.bfloat16)
        else:
            out[key] = tensor
    return out


def load_conditional_transformer(dtype: torch.dtype = DTYPE) -> Ideogram4Transformer:
    """Load the conditional DiT branch with FP8 weights materialized to bf16."""
    import os

    from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

    config = Ideogram4Config()

    # DEBUG (hang triage, revertible): IDEOGRAM4_DEBUG_LAYERS shrinks the block
    # count for a cheap compile; IDEOGRAM4_DEBUG_RANDOM=1 skips the HF download +
    # single-threaded FP8->bf16 materialize (~30 min) and random-inits instead.
    # A completion-queue device hang is structural (op sequence + shapes, not
    # weight values), so random weights reproduce it while iterating fast.
    debug_layers = os.environ.get("IDEOGRAM4_DEBUG_LAYERS")
    if debug_layers:
        config.num_layers = int(debug_layers)
    if os.environ.get("IDEOGRAM4_DEBUG_RANDOM") == "1":
        model = Ideogram4Transformer(config)
        model.to(dtype=dtype)
        model.eval()
        return model

    state_dict = _load_sharded_state_dict(
        REPO_ID,
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    state_dict_bf16 = materialize_fp8_state_dict_to_bf16(state_dict)
    model = Ideogram4Transformer(config)
    model.load_state_dict(state_dict_bf16, strict=True)
    model.to(dtype=dtype)
    model.eval()
    return model


class Ideogram4TransformerWrapper(nn.Module):
    """Thin wrapper so the test runner can call forward with keyword tensors."""

    def __init__(self, transformer: Ideogram4Transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )


def build_synthetic_transformer_inputs(
    batch_size: int = 1, dtype: torch.dtype = DTYPE
) -> dict[str, torch.Tensor]:
    """Synthetic packed-sequence inputs at 512x512 resolution."""
    from ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR

    llm_features = torch.randn(batch_size, TOTAL_SEQ_LEN, LLM_FEATURES_DIM, dtype=dtype)
    x = torch.randn(batch_size, TOTAL_SEQ_LEN, IN_CHANNELS, dtype=dtype)
    t = torch.full((batch_size,), 0.5, dtype=dtype)

    position_ids = torch.zeros(batch_size, TOTAL_SEQ_LEN, 3, dtype=torch.long)
    # Text positions: t=0, h=0, w=token_index
    for i in range(MAX_TEXT_TOKENS):
        position_ids[:, i, 0] = 0
        position_ids[:, i, 1] = 0
        position_ids[:, i, 2] = i
    # Image positions: offset grid (simplified — matches pipeline layout)
    idx = 0
    for h in range(GRID_H):
        for w in range(GRID_W):
            pos = MAX_TEXT_TOKENS + idx
            position_ids[:, pos, 0] = 0
            position_ids[:, pos, 1] = h
            position_ids[:, pos, 2] = w
            idx += 1

    segment_ids = torch.zeros(batch_size, TOTAL_SEQ_LEN, dtype=torch.long)
    indicator = torch.full(
        (batch_size, TOTAL_SEQ_LEN), OUTPUT_IMAGE_INDICATOR, dtype=torch.long
    )
    indicator[:, :MAX_TEXT_TOKENS] = LLM_TOKEN_INDICATOR

    return {
        "llm_features": llm_features,
        "x": x,
        "t": t,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
    }


# ---------------------------------------------------------------------------
# Tensor-parallel shard specs (Megatron column->row).
#
# Ideogram 4 attention CANNOT be head-parallel sharded: num_heads=18 does not
# divide 4, and the qkv projection is a single fused Linear(emb -> 3*emb) whose
# output reshapes to (3, heads, head_dim) -- a flat column shard slices across
# the q|k|v and head boundaries. So attention (qkv, o) stays REPLICATED and only
# the SwiGLU MLP is sharded (w1/w3 column-parallel, w2 row-parallel).
#
# NOTE: llm_cond_proj (53248 -> emb) is deliberately NOT row-sharded. Doing so
# back-propagates the shard into the preceding llm_cond_norm, turning its
# reduction over the 53248 dim into a distributed rms_norm_pre_all_gather whose
# circular buffer (3.36 MB) overflows Blackhole L1 (1.5 MB max). It is only
# ~2.7% of the model, so replicating it costs little per chip.
#
# Activations stay replicated end-to-end, so blocks chain with no reshards and
# the only collective is one all-reduce per row-parallel (w2) matmul. See
# sharding_analysis.md for the full derivation and CCL accounting.
# ---------------------------------------------------------------------------

MESH_SHAPES = {8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def _add_shard_spec(specs: dict, param, spec: tuple) -> None:
    """Register a partition spec only for real parameters (skip None)."""
    if param is not None:
        specs[param] = spec


def _shard_linear(specs: dict, linear, spec: tuple) -> None:
    """Shard a linear's weight per `spec`; bias along the first axis or replicated."""
    if linear is None:
        return
    _add_shard_spec(specs, linear.weight, spec)
    bias_spec = ("model",) if spec[0] == "model" else (None,)
    _add_shard_spec(specs, getattr(linear, "bias", None), bias_spec)


def shard_transformer_specs(transformer) -> dict:
    """Tensor-parallel shard specs for Ideogram4Transformer (Option A, mesh 1x4).

    SwiGLU MLP Megatron-sharded 4-way; attention, adaln, norms, llm_cond_proj and
    the small embedders replicated (unmarked). ~53% weight per chip, 34 all-reduces.
    """
    specs: dict = {}

    for block in transformer.layers:
        ff = block.feed_forward
        # SwiGLU: up (w1) and gate (w3) column-parallel; down (w2) row-parallel.
        _shard_linear(specs, ff.w1, ("model", None))
        _shard_linear(specs, ff.w3, ("model", None))
        _shard_linear(specs, ff.w2, (None, "model"))

    return specs


# ---------------------------------------------------------------------------
# VAE decoder component.
#
# `Ideogram4Pipeline._decode` calls `self.autoencoder.decoder(z)` directly, so
# the decoder submodule - not the full AutoEncoder - is the compilable unit and
# the one worth comparing against CPU. Its input is the *unpatched* AE latent:
# the pipeline undoes the DiT's 2x2 patching before handing `z` over.
# ---------------------------------------------------------------------------

VAE_WEIGHTS_FILENAME = "vae/diffusion_pytorch_model.safetensors"
# ch_mult has 4 entries -> three 2x upsamples -> 8x spatial scale.
VAE_SPATIAL_SCALE = 8
VAE_DEFAULT_RESOLUTION = 512


def load_vae_decoder(dtype: torch.dtype = DTYPE):
    """Load the Ideogram 4 autoencoder and return its decoder submodule."""
    from ideogram4.pipeline_ideogram4 import _load_autoencoder

    weights_path = hf_hub_download(repo_id=REPO_ID, filename=VAE_WEIGHTS_FILENAME)
    autoencoder = _load_autoencoder(weights_path, torch.device("cpu"), dtype)
    return autoencoder.decoder.eval()


def build_vae_decoder_inputs(
    dtype: torch.dtype = DTYPE, resolution: int = VAE_DEFAULT_RESOLUTION
) -> list:
    """Synthetic unpatched AE latent, matching what `_decode` feeds the decoder."""
    from ideogram4.autoencoder import AutoEncoderParams

    params = AutoEncoderParams()
    side = resolution // VAE_SPATIAL_SCALE
    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(
        1, params.z_channels, side, side, dtype=torch.float32, generator=generator
    )
    return [latent.to(dtype)]
