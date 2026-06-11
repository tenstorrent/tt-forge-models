# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel shard specs for the BAGEL Qwen2 MoT backbone (Megatron-1D).

The backbone is a Qwen2-7B-scale Mixture-of-Transformers:
  hidden=3584, intermediate=18944, num_attention_heads=28, num_key_value_heads=4, layers=28.

28 heads is not divisible by 8, so 8-way TP is invalid. The largest valid degree on the
8-chip Wormhole fabric is **4-way** (28 q-heads / 4 = 7 per shard; 4 kv-heads / 4 = 1 per
shard — GQA stays consistent). Mesh is Megatron-1D ``(None, "model")``.

Column-parallel  ("model", None): shard out_features (dim 0) — q/k/v/gate/up projections.
Row-parallel     (None, "model"): shard in_features  (dim 1) — o_proj / down_proj.
Replicate        (None,)        : norms, embeddings, biases, 1-D params.

Both the understanding ("und") and generation ("gen", ``*_moe_gen``) expert weights are
sharded identically by matching on the projection suffix.
"""

from __future__ import annotations

MESH_NAMES = (None, "model")
# Only 2 and 4 are head-divisible on this fabric; 4-way is preferred (fits 12 GiB/chip).
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4)}

_COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW_PARALLEL = ("o_proj", "down_proj")


def get_mesh_shape(num_devices: int):
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count for BAGEL MoT backbone: {num_devices}. "
            f"Valid (head-divisible) degrees: {sorted(MESH_SHAPES)} (8-way invalid: 28 heads)."
        )
    return MESH_SHAPES[num_devices], MESH_NAMES


def shard_backbone_specs(model) -> dict:
    """Map each backbone parameter to a Megatron-1D partition spec.

    ``model`` may be the BagelTextBackbone wrapper or the bare Qwen2Model. Linear weights are
    [out_features, in_features]; column-parallel shards dim 0, row-parallel shards dim 1.
    """
    backbone = getattr(model, "backbone", model)
    specs = {}
    for name, param in backbone.named_parameters():
        leaf = (
            name.split(".")[-2] if "." in name else name
        )  # e.g. "q_proj" from "...q_proj.weight"
        is_weight = name.endswith(".weight")
        is_bias = name.endswith(".bias")
        # strip the moe_gen suffix so und/gen experts share the same rule
        base_leaf = leaf.replace("_moe_gen", "")

        if base_leaf in _COLUMN_PARALLEL and is_weight:
            # column-parallel: shard out_features (dim 0)
            specs[param] = ("model", None)
        elif base_leaf in _COLUMN_PARALLEL and is_bias:
            # the matching bias is per-output -> shard along the same (model) axis
            specs[param] = ("model",)
        elif base_leaf in _ROW_PARALLEL and is_weight:
            # row-parallel: shard in_features (dim 1); bias (if any) stays replicated
            specs[param] = (None, "model")
        else:
            # norms, embed_tokens, row-parallel biases, rotary buffers -> replicate
            specs[param] = (None,) * param.dim()
    return specs
