# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Megatron column->row tensor-parallel shard specs for the SD3 MMDiT denoiser
(``SD3Transformer2DModel``).

Mesh axes: ``("batch", "model")``. Only ``"model"`` is a real shard axis; the
``"batch"`` (data-parallel) axis stays replicated, so any dim that must not be
sharded is marked ``None`` rather than ``"batch"``.

Each ``JointTransformerBlock`` has a joint attention (``attn``) that mixes an
image stream (``to_q/k/v`` -> ``to_out``) and a text stream (``add_{q,k,v}_proj``
-> ``to_add_out``), plus an MLP (``ff``) and a text MLP (``ff_context``). SD3
uses ``bias=True`` everywhere. The last block is ``context_pre_only=True``: it
has no ``to_add_out`` and no ``ff_context`` (guarded below).

Column-parallel (Q/K/V, FF up):  weight ("model", None), bias ("model",)
Row-parallel    (out, FF down):  weight (None, "model"), bias replicated (None,)
patch_embed / pos_embed / proj_out / norms / time-text embed: replicated.
"""


def _column(specs, linear):
    """Mark a Linear column-parallel (shard output dim on 'model')."""
    if linear is None:
        return
    specs[linear.weight] = ("model", None)
    if getattr(linear, "bias", None) is not None:
        specs[linear.bias] = ("model",)


def _row(specs, linear):
    """Mark a Linear row-parallel (shard input dim on 'model'); bias replicated."""
    if linear is None:
        return
    specs[linear.weight] = (None, "model")
    if getattr(linear, "bias", None) is not None:
        specs[linear.bias] = (None,)


def shard_transformer_specs(transformer) -> dict:
    """Build a ``{tensor: partition_spec}`` dict for ``SD3Transformer2DModel``.

    Sharding is valid only when ``num_attention_heads`` divides the mesh model
    axis (24 heads % 2 == 0 on n300). The caller asserts this.
    """
    specs: dict = {}

    for block in transformer.transformer_blocks:
        attn = block.attn

        # Image-stream Q/K/V (column-parallel).
        _column(specs, attn.to_q)
        _column(specs, attn.to_k)
        _column(specs, attn.to_v)

        # Text-stream Q/K/V (column-parallel) — present unless context_pre_only.
        _column(specs, getattr(attn, "add_q_proj", None))
        _column(specs, getattr(attn, "add_k_proj", None))
        _column(specs, getattr(attn, "add_v_proj", None))

        # Image-stream output projection (row-parallel).
        _row(specs, attn.to_out[0])

        # Text-stream output projection (row-parallel) — None on the last block.
        if not attn.context_pre_only and getattr(attn, "to_add_out", None) is not None:
            _row(specs, attn.to_add_out)

        # Image MLP (GELU): net[0].proj column-parallel, net[2] row-parallel.
        _column(specs, block.ff.net[0].proj)
        _row(specs, block.ff.net[2])

        # Text MLP — absent on the last (context_pre_only) block.
        if getattr(block, "ff_context", None) is not None:
            _column(specs, block.ff_context.net[0].proj)
            _row(specs, block.ff_context.net[2])

    return specs
