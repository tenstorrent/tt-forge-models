# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel shard specifications for the Stable Diffusion 3 Medium MMDiT.

SD3-medium's denoiser is an ``SD3Transformer2DModel`` — 24 stacked
``JointTransformerBlock`` (MMDiT) blocks operating jointly on the image latent
stream and the text (context) stream, ``dim`` 1536 with 24 attention heads.

This module implements **Megatron-style 1-D tensor parallelism** over a
``(None, "model")`` mesh, the same pattern used by the ``fibo`` / ``mochi`` /
``hunyuan_video`` DiT loaders:

- Attention / FFN are sharded **column → row** (Megatron): the first matmul of
  each pair shards its output dim (``("model", None)``) and the second its
  contracting dim (``(None, "model")``), producing one all-reduce per pair.
- Adaptive-layernorm modulation linears (``norm1.linear`` / ``norm1_context.linear``)
  are **row-parallel** → all-reduced back to a replicated tensor: the modulation
  output is immediately chunked into shift/scale/gate and broadcast over the
  (replicated) residual stream, so a replicated output keeps that local.
- Everything else (patch / pos / time embedders, LayerNorm weights, the final
  ``proj_out`` and biases after a row-parallel matmul) is **replicated**.

SD3-medium specifics handled here:
- ``use_dual_attention=False`` → ``attn2`` is ``None`` (no second attention).
- The **last block** has ``context_pre_only=True``: its context stream is not
  updated after attention, so ``attn.to_add_out``, ``ff_context`` and
  ``norm2_context`` are ``None`` and ``norm1_context`` is an
  ``AdaLayerNormContinuous``. Every context-side linear is therefore guarded.

24 heads / ``dim`` 1536 are divisible by 1 and 2, so the mesh degrees below are
valid for n150 (1) and n300 (2).
"""

from typing import Dict, Tuple

# Pure Megatron-1D: only the "model" axis shards. The leading axis is unnamed
# (no data parallelism) to match the mochi / fibo DiT pattern.
MESH_NAMES: Tuple = (None, "model")

# num_devices -> mesh_shape. The TP degree is the second ("model") axis.
MESH_SHAPES: Dict[int, Tuple[int, int]] = {
    1: (1, 1),
    2: (1, 2),
}


def get_mesh_shape(num_devices: int) -> Tuple[Tuple[int, int], Tuple]:
    """Return ``(mesh_shape, mesh_names)`` for ``num_devices`` chips.

    Raises:
        ValueError: if ``num_devices`` is not a supported SD3 TP degree.
    """
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
    """Shard one ``JointTransformerBlock`` (MMDiT joint image/text block)."""
    # Adaptive-layernorm modulation: row-parallel -> replicated, chunked output.
    # norm1 is always AdaLayerNormZero; norm1_context is AdaLayerNormZero except
    # on the context_pre_only block where it is AdaLayerNormContinuous — both
    # expose a `.linear`.
    _shard_linear_row(specs, getattr(block.norm1, "linear", None))
    _shard_linear_row(specs, getattr(block.norm1_context, "linear", None))

    attn = block.attn
    # Image-stream Q/K/V — column-parallel by heads.
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)
    # Text-stream (added) Q/K/V — column-parallel by heads (may be absent).
    _shard_linear_column(specs, getattr(attn, "add_q_proj", None))
    _shard_linear_column(specs, getattr(attn, "add_k_proj", None))
    _shard_linear_column(specs, getattr(attn, "add_v_proj", None))
    # Output projections — row-parallel. to_add_out is None on context_pre_only.
    _shard_linear_row(specs, attn.to_out[0])
    _shard_linear_row(specs, getattr(attn, "to_add_out", None))

    # FeedForward for the image stream, and the context stream when present.
    _shard_feed_forward(specs, block.ff)
    _shard_feed_forward(specs, getattr(block, "ff_context", None))


def build_shard_spec(model) -> dict:
    """Build the SD3 MMDiT shard spec.

    Args:
        model: the ``SD3Transformer2DModel`` returned by ``ModelLoader.load_model``
            (or a thin wrapper exposing it at ``.model`` / ``.transformer``).

    Returns:
        dict mapping each sharded ``torch.nn.Parameter`` to its partition spec.
        Parameters not present in the dict are replicated across the mesh.
    """
    transformer = model
    if not hasattr(transformer, "transformer_blocks"):
        transformer = getattr(model, "transformer", None) or getattr(
            model, "model", model
        )

    specs: dict = {}
    for block in transformer.transformer_blocks:
        _shard_joint_block(specs, block)

    return specs
