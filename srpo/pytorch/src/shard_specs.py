# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel shard specifications for the SRPO (tencent/SRPO) transformer.

SRPO is a FLUX.1-dev fine-tune (arXiv 2509.06942); its checkpoint is a drop-in
``diffusers.FluxTransformer2DModel`` state dict (~12B params). Like FLUX.1-dev
and FIBO it runs out of DRAM on a single chip: the FFN intermediate activation
(``inner_dim`` 3072 -> ``mlp`` 12288), the attention score matrices, and the
adaptive-layernorm modulation projections together exceed one chip's budget.

This module implements **Megatron-style 1-D tensor parallelism** over a
``(None, "model")`` mesh, following the established DiT pattern (mochi /
hunyuan_video / fibo):

- Attention / FFN are sharded **column -> row** (one all-reduce per pair).
- Adaptive-layernorm modulation linears (``norm1.linear`` etc.) are sharded
  **row-parallel** so their output is all-reduced back to *replicated* — the
  chunked shift/scale/gate then stays local, matching the HunyuanVideo loader's
  adaLN handling.
- Everything else (embedders, time/text projections, norms, final ``proj_out``)
  is **replicated**.

The FLUX DiT module structure
(``diffusers.models.transformers.transformer_flux.FluxTransformer2DModel``):

    FluxTransformer2DModel
      ├─ x_embedder / context_embedder / time_text_embed   (replicated)
      ├─ transformer_blocks[19]   (FluxTransformerBlock, MMDiT)
      │    ├─ norm1.linear / norm1_context.linear   (AdaLayerNormZero -> row)
      │    ├─ attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj}  (col)
      │    ├─ attn.{to_out[0],to_add_out}                             (row)
      │    └─ ff / ff_context  (FeedForward: net[0].proj col, net[2] row)
      ├─ single_transformer_blocks[38]   (FluxSingleTransformerBlock)
      │    ├─ norm.linear   (AdaLayerNormZeroSingle -> row)
      │    ├─ attn.{to_q,to_k,to_v}   (col; pre_only, no to_out)
      │    ├─ proj_mlp                (col)
      │    └─ proj_out                (row; consumes cat(attn, mlp) = 15360)
      └─ norm_out.linear (AdaLayerNormContinuous -> row) / proj_out (replicated)
"""

from typing import Dict, Tuple

# Pure Megatron-1D: only the "model" axis is sharded. The leading axis is
# unnamed (no data parallelism) to match the mochi / bagel / fibo DiT pattern.
MESH_NAMES: Tuple = (None, "model")

# num_devices -> mesh_shape. The TP degree is the second ("model") axis.
# FLUX/SRPO has 24 attention heads and inner_dim 3072 / mlp 12288 /
# single-block concat 15360 — all divisible by 1, 2, 4 and 8.
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
            f"SRPO tensor-parallel supports device counts "
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
    _add(specs, linear.weight, ("model", None))
    if has_bias and getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, ("model",))


def _shard_linear_row(specs: dict, linear) -> None:
    """Row-parallel: shard contracting dim of weight ``[out, in]`` on ``model``.

    The output is all-reduced back to a replicated tensor, so the bias (added
    after the reduction) is replicated.
    """
    _add(specs, linear.weight, (None, "model"))
    if getattr(linear, "bias", None) is not None:
        _add(specs, linear.bias, (None,))


def _shard_feed_forward(specs: dict, ff) -> None:
    """diffusers ``FeedForward``: ``net[0]`` is GELU(.proj), ``net[2]`` Linear."""
    _shard_linear_column(specs, ff.net[0].proj)
    _shard_linear_row(specs, ff.net[2])


def _shard_mmdit_block(specs: dict, block) -> None:
    """Shard one ``FluxTransformerBlock`` (joint image/text MMDiT block)."""
    # Adaptive-layernorm modulation: row-parallel -> replicated, chunked output.
    _shard_linear_row(specs, block.norm1.linear)
    _shard_linear_row(specs, block.norm1_context.linear)

    attn = block.attn
    # Image-stream Q/K/V — column-parallel by heads.
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)
    # Text-stream (added) Q/K/V — column-parallel by heads.
    _shard_linear_column(specs, attn.add_q_proj)
    _shard_linear_column(specs, attn.add_k_proj)
    _shard_linear_column(specs, attn.add_v_proj)
    # Output projections — row-parallel.
    _shard_linear_row(specs, attn.to_out[0])
    _shard_linear_row(specs, attn.to_add_out)

    # FeedForward for both the image and the text (context) stream.
    _shard_feed_forward(specs, block.ff)
    _shard_feed_forward(specs, block.ff_context)


def _shard_single_block(specs: dict, block) -> None:
    """Shard one ``FluxSingleTransformerBlock`` (fused attn+MLP block)."""
    # Adaptive-layernorm modulation: row-parallel -> replicated, chunked output.
    _shard_linear_row(specs, block.norm.linear)

    attn = block.attn
    # pre_only=True attention: only Q/K/V projections exist (no to_out). Sharded
    # column-parallel so the attention output is sharded on the model axis along
    # its last dim — matching proj_mlp's output for the downstream concat.
    _shard_linear_column(specs, attn.to_q)
    _shard_linear_column(specs, attn.to_k)
    _shard_linear_column(specs, attn.to_v)

    # MLP project-in (column) and the fused project-out (row). proj_out consumes
    # cat([attn_output(3072), mlp(12288)], dim=-1) = 15360, both halves sharded
    # on model along the concat dim, so the row-parallel contraction is clean.
    _shard_linear_column(specs, block.proj_mlp)
    _shard_linear_row(specs, block.proj_out)


def build_shard_spec(model) -> dict:
    """Build the SRPO/FLUX transformer shard spec.

    Args:
        model: the ``FluxTransformer2DModel`` returned by ``ModelLoader`` (or a
            wrapper exposing it at ``.transformer``).

    Returns:
        dict mapping each sharded ``torch.nn.Parameter`` to its partition spec.
        Parameters not present in the dict are replicated across the mesh.
    """
    transformer = getattr(model, "transformer", model)

    specs: dict = {}
    for block in transformer.transformer_blocks:
        _shard_mmdit_block(specs, block)
    for block in transformer.single_transformer_blocks:
        _shard_single_block(specs, block)

    # Final adaptive-layernorm before proj_out — row-parallel like the rest.
    norm_out = getattr(transformer, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        _shard_linear_row(specs, norm_out.linear)

    return specs
