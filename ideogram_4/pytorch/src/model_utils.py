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

from ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

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
    config = Ideogram4Config()
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
