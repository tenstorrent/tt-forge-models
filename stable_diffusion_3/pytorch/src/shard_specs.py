# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel shard specifications for the SD3 Medium MMDiT.

``SD3Transformer2DModel`` is a 24-layer MMDiT (``JointTransformerBlock`` stack,
24 attention heads). On an n300 (2 chips) the single-chip baseline already fits,
so TP here is a *baseline-setting* step, not an OOM rescue — it gives perf-tuning
a ~2x-compute starting point.

This module implements **Megatron-style 1-D tensor parallelism** over a
``(None, "model")`` mesh, following the FIBO / mochi / hunyuan_video DiT pattern:

- Attention / FeedForward are sharded **column → row**: the Q/K/V projections
  (image stream ``to_{q,k,v}`` and text stream ``add_{q,k,v}_proj``) and the FFN
  project-in shard their output dim ``("model", None)``; the output projections
  (``to_out[0]`` / ``to_add_out``) and the FFN project-out shard their
  contracting dim ``(None, "model")`` — one all-reduce per pair.
- Adaptive-layernorm modulation linears (``norm1.linear`` / ``norm1_context.linear``
  / final ``norm_out.linear``) are sharded **row-parallel** so their output is
  all-reduced back to a *replicated* tensor; the modulation is immediately
  chunked into shift/scale/gate and broadcast over the replicated residual
  stream, so a replicated output keeps that slice-and-modulate local.
- Everything else (patch/pos/timestep/context embedders, norms, ``proj_out``,
  and biases following a row-parallel matmul) is **replicated**.

Block structure (``diffusers.models.transformers.transformer_sd3``):

    SD3Transformer2DModel
      ├─ pos_embed / time_text_embed / context_embedder      (replicated)
      ├─ transformer_blocks[24]  (JointTransformerBlock, MMDiT)
      │    ├─ norm1.linear / norm1_context.linear   (AdaLayerNormZero -> row)
      │    ├─ attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj}  (col)
      │    ├─ attn.{to_out[0], to_add_out}                            (row)
      │    └─ ff / ff_context  (FeedForward: net[0].proj col, net[2] row)
      │   (the final block has context_pre_only=True: no ff_context / to_add_out)
      └─ norm_out.linear (AdaLayerNormContinuous -> row) / proj_out (replicated)
"""

from typing import Dict, Tuple

# Pure Megatron-1D: only the "model" axis shards. Leading axis unnamed (no DP),
# matching the mochi / FIBO DiT pattern.
MESH_NAMES: Tuple = (None, "model")

# num_devices -> mesh_shape. The TP degree is the second ("model") axis. SD3 has
# 24 attention heads (div by 1, 2, 3, 4, 6, 8); the degrees below all divide it.
MESH_SHAPES: Dict[int, Tuple[int, int]] = {
    1: (1, 1),
    2: (1, 2),
    4: (1, 4),
    8: (1, 8),
}


def get_mesh_shape(num_devices: int) -> Tuple[Tuple[int, int], Tuple]:
    """Return ``(mesh_shape, mesh_names)`` for ``num_devices`` chips."""
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"SD3 tensor-parallel supports device counts "
            f"{sorted(MESH_SHAPES)}, got {num_devices}."
        )
    return MESH_SHAPES[num_devices], MESH_NAMES


def _add(specs: dict, tensor, spec: Tuple) -> None:
    """Record ``tensor -> spec`` if the tensor exists and rank matches."""
    if tensor is None:
        return
    if tensor.dim() != len(spec):
        raise ValueError(
            f"Shard spec {spec} has rank {len(spec)} but tensor has rank "
            f"{tensor.dim()} (shape {tuple(tensor.shape)})."
        )
    specs[tensor] = spec


def _shard_linear_column(specs: dict, linear, *, has_bias: bool = True) -> None:
    """Column-parallel: shard output dim of weight ``[out, in]`` on ``model``."""
    if linear is None:
        return
    _add(specs, linear.weight, ("model", None))
    if has_bias and getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, ("model",))


def _shard_linear_row(specs: dict, linear) -> None:
    """Row-parallel: shard contracting dim of weight ``[out, in]`` on ``model``.

    The output is all-reduced back to a replicated tensor, so the bias (added
    after the reduction) is replicated.
    """
    if linear is None:
        return
    _add(specs, linear.weight, (None, "model"))
    if getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, (None,))


def _shard_feed_forward(specs: dict, ff) -> None:
    """diffusers ``FeedForward``: ``net[0]`` is GELU(.proj), ``net[2]`` Linear."""
    if ff is None:
        return
    _shard_linear_column(specs, ff.net[0].proj)
    _shard_linear_row(specs, ff.net[2])


def _shard_joint_block(specs: dict, block) -> None:
    """Shard one ``JointTransformerBlock`` (joint image/text MMDiT block)."""
    # Adaptive-layernorm modulation: row-parallel -> replicated, chunked output.
    _shard_linear_row(specs, getattr(block.norm1, "linear", None))
    norm1_context = getattr(block, "norm1_context", None)
    if norm1_context is not None:
        _shard_linear_row(specs, getattr(norm1_context, "linear", None))

    attn = block.attn
    # Image-stream Q/K/V — column-parallel by heads.
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)
    # Text-stream (added) Q/K/V — column-parallel by heads.
    _shard_linear_column(specs, getattr(attn, "add_q_proj", None))
    _shard_linear_column(specs, getattr(attn, "add_k_proj", None))
    _shard_linear_column(specs, getattr(attn, "add_v_proj", None))
    # Output projections — row-parallel. (to_add_out is absent on the final,
    # context_pre_only block.)
    _shard_linear_row(specs, attn.to_out[0])
    _shard_linear_row(specs, getattr(attn, "to_add_out", None))

    # FeedForward for the image stream, and the text stream when present.
    _shard_feed_forward(specs, block.ff)
    _shard_feed_forward(specs, getattr(block, "ff_context", None))


def build_shard_spec(model) -> dict:
    """Build the SD3 transformer shard spec.

    Args:
        model: the wrapper returned by the tester (the DiT lives at
            ``model.model``/``model.transformer``) or the bare
            ``SD3Transformer2DModel``.

    Returns:
        dict mapping each sharded ``torch.nn.Parameter`` to its partition spec.
        Parameters not present in the dict are replicated across the mesh.
    """
    transformer = model
    for attr in ("model", "transformer"):
        if hasattr(transformer, attr) and hasattr(
            getattr(transformer, attr), "transformer_blocks"
        ):
            transformer = getattr(transformer, attr)
            break

    specs: dict = {}
    for block in transformer.transformer_blocks:
        _shard_joint_block(specs, block)

    # Final adaptive-layernorm before proj_out — row-parallel like the rest.
    norm_out = getattr(transformer, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        _shard_linear_row(specs, norm_out.linear)

    return specs
