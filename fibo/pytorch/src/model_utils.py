# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FIBO (briaai/FIBO) loading, input construction and
tensor-parallel sharding.

FIBO is BRIA AI's ~8B-parameter DiT-based, flow-matching text-to-image model.
It uses ``SmolLM3-3B`` as the text encoder, a ``Wan`` VAE, and a Flux-style
MMDiT denoiser (``BriaFiboTransformer2DModel``) with a novel "DimFusion"
conditioning scheme that injects one SmolLM3 hidden-state layer into each
transformer block. The full pipeline is exposed as ``BriaFiboPipeline`` in
diffusers (>= 0.38).

Bringup focuses on the **denoiser** (``BriaFiboTransformer2DModel``) — the
compute-dominant component that must run on device. The denoiser alone is
~16.5 GB in bf16, which does not fit a single Wormhole chip (12 GB), so on
n300 it is sharded tensor-parallel across both chips (see ``fibo_shard_specs``).

``load_inputs`` builds synthetic denoiser inputs at FIBO's native 1024x1024
resolution (latent sequence length 64*64 = 4096) with correct shapes/dtypes,
mirroring exactly what ``BriaFiboPipeline.__call__`` feeds the transformer at
one denoising step. This keeps the per-component bringup memory-bounded to the
transformer only (the 3B text encoder is not loaded for the denoiser test) and
avoids pinning to a brittle captured call signature.

Reference: https://huggingface.co/briaai/FIBO  (arXiv:2511.06876)
"""

import os
from typing import Any, Dict, List, Optional

import torch

# FIBO denoiser config (transformer/config.json on briaai/FIBO).
IN_CHANNELS = 48  # latent channels fed to the transformer (do_patching=False)
NUM_ATTENTION_HEADS = 24
ATTENTION_HEAD_DIM = 128
INNER_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 3072
JOINT_ATTENTION_DIM = 4096  # encoder_hidden_states feature dim
TEXT_ENCODER_DIM = 2048  # per-layer SmolLM3 hidden size
NUM_LAYERS = 8  # dual-stream MMDiT blocks
NUM_SINGLE_LAYERS = 38  # single-stream blocks
TOTAL_BLOCKS = NUM_LAYERS + NUM_SINGLE_LAYERS  # 46 text-encoder layers consumed

# Native generation geometry: default_sample_size (64) * vae_scale_factor (16)
# = 1024px. do_patching=False => latent grid is 1024/16 = 64 per side.
NATIVE_RESOLUTION = 1024
VAE_SCALE_FACTOR = 16
LATENT_GRID = NATIVE_RESOLUTION // VAE_SCALE_FACTOR  # 64
LATENT_SEQ_LEN = LATENT_GRID * LATENT_GRID  # 4096

# Prompt sequence length used for the synthetic encoder/caption tensors. FIBO
# pads to the longest tokenized structured-JSON caption; 128 is a representative
# length that keeps the joint attention sequence (128 + 4096) tractable.
PROMPT_SEQ_LEN = 128


def build_fibo_inputs(
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Construct synthetic denoiser inputs at FIBO's native 1024x1024 resolution.

    Shapes mirror one ``BriaFiboPipeline`` denoising step (without classifier-
    free-guidance batch doubling — a single forward pass is sufficient for
    component bringup; resolution, not CFG batch, is what must stay native).

    Returns a kwargs dict the runner feeds as ``model(**inputs)``. ``return_dict``
    is False so the transformer returns a plain ``(sample,)`` tuple the comparison
    harness can tree-map over.
    """
    # Native geometry by default. FIBO_LATENT_GRID / FIBO_PROMPT_SEQ_LEN env
    # overrides exist only to smoke-test the device sharding/compile path at
    # reduced cost; the committed default is full 1024x1024 (grid 64).
    grid = int(os.environ.get("FIBO_LATENT_GRID", LATENT_GRID))
    latent_seq = grid * grid
    prompt_seq = int(os.environ.get("FIBO_PROMPT_SEQ_LEN", PROMPT_SEQ_LEN))

    g = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(dtype)

    hidden_states = randn(batch_size, latent_seq, IN_CHANNELS)
    encoder_hidden_states = randn(batch_size, prompt_seq, JOINT_ATTENTION_DIM)
    text_encoder_layers: List[torch.Tensor] = [
        randn(batch_size, prompt_seq, TEXT_ENCODER_DIM) for _ in range(TOTAL_BLOCKS)
    ]

    # Rotary position ids: [seq, 3] (time, height, width axes -> axes_dims_rope).
    txt_ids = torch.zeros(prompt_seq, 3, dtype=dtype)
    img_ids = torch.zeros(latent_seq, 3, dtype=dtype)
    row = torch.arange(grid)
    img_ids[:, 1] = row[:, None].expand(grid, grid).reshape(-1).to(dtype)
    img_ids[:, 2] = row[None, :].expand(grid, grid).reshape(-1).to(dtype)

    # Mid-trajectory timestep (num_train_timesteps=1000).
    timestep = torch.full((batch_size,), 500.0, dtype=dtype)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "text_encoder_layers": text_encoder_layers,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        # guidance_embeds=False on this checkpoint -> no guidance tensor.
        "guidance": None,
        # No padding in the synthetic prompt => an all-valid additive mask is a
        # no-op, so omit it (numerically identical to the pipeline's mask) and
        # save a [seq, seq] device tensor.
        "joint_attention_kwargs": {},
        "return_dict": False,
    }


def _patched_single_block_forward(self, hidden_states, temb, image_rotary_emb=None,
                                   joint_attention_kwargs=None):
    """Single-block forward that avoids the wide ``cat([attn, mlp])`` concat.

    The stock ``BriaFiboSingleTransformerBlock`` computes
    ``proj_out(cat([attn_output(3072), mlp_hidden(12288)], dim=-1))`` — a
    15360-wide concat whose ttnn CB page (~2.27 MB) exceeds Wormhole per-core
    L1 (~1.33 MB) and cannot lower (TT_FATAL in concat_program_factory). Since
    ``proj_out`` is linear, ``proj_out(cat([a, m])) == proj_out_a(a) + proj_out_m(m)``
    where ``proj_out_a``/``proj_out_m`` are the column slices of ``proj_out``
    (the +bias kept on the mlp half). This is numerically identical and removes
    the concat entirely, so the op fits L1. ``apply_single_block_rewrite``
    installs the split projections this method consumes.
    """
    import torch.nn.functional as F

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    proj = F.linear(attn_output, self.proj_out_attn.weight) + F.linear(
        mlp_hidden_states, self.proj_out_mlp.weight, self.proj_out_mlp.bias
    )
    hidden_states = residual + gate.unsqueeze(1) * proj
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def apply_single_block_rewrite(model: torch.nn.Module) -> int:
    """Split each single block's ``proj_out`` into attn/mlp halves and install
    the concat-free forward. Returns the number of blocks rewritten.

    ``proj_out`` maps ``[attn_dim + mlp_hidden] -> dim``; its weight columns
    ``[:, :attn_dim]`` act on the attention output and ``[:, attn_dim:]`` on the
    mlp hidden, so two ``nn.Linear`` layers reproduce it exactly.
    """
    import types

    from torch import nn

    count = 0
    for blk in model.single_transformer_blocks:
        if hasattr(blk, "proj_out_attn"):
            continue
        W = blk.proj_out.weight  # [dim, attn_dim + mlp_hidden]
        b = blk.proj_out.bias
        out_dim, in_dim = W.shape
        attn_dim = blk.attn.out_dim  # inner_dim (3072)
        mlp_dim = in_dim - attn_dim

        proj_out_attn = nn.Linear(attn_dim, out_dim, bias=False)
        proj_out_mlp = nn.Linear(mlp_dim, out_dim, bias=b is not None)
        with torch.no_grad():
            proj_out_attn.weight.copy_(W[:, :attn_dim])
            proj_out_mlp.weight.copy_(W[:, attn_dim:])
            if b is not None:
                proj_out_mlp.bias.copy_(b)
        proj_out_attn = proj_out_attn.to(W.dtype)
        proj_out_mlp = proj_out_mlp.to(W.dtype)
        for p in proj_out_attn.parameters():
            p.requires_grad = False
        for p in proj_out_mlp.parameters():
            p.requires_grad = False

        blk.proj_out_attn = proj_out_attn
        blk.proj_out_mlp = proj_out_mlp
        del blk.proj_out  # drop the fused projection so its weight isn't resident
        blk.forward = types.MethodType(_patched_single_block_forward, blk)
        count += 1
    return count


def fibo_shard_specs(model: torch.nn.Module, model_axis: str = "model") -> Dict:
    """Megatron-style tensor-parallel shard specs for the FIBO denoiser.

    Column-parallel on every attention/MLP input projection (shard output
    features along ``model_axis``), row-parallel on every output projection
    (shard input features), on a ("batch", model_axis) mesh. Heads (24) are
    divisible by 2, so this maps cleanly onto n300's two Wormhole chips.

    Only ``.weight`` tensors are annotated; GSPMD/Shardy propagation infers the
    matching sharding for biases, RMSNorm scales and the elementwise adds.
    """
    col = (model_axis, None)  # shard output dim (dim 0 of [out, in] weight)
    row = (None, model_axis)  # shard input dim (dim 1)
    specs: Dict[torch.Tensor, tuple] = {}

    # Dual-stream MMDiT blocks: joint image+text attention with added KV proj.
    for blk in model.transformer_blocks:
        attn = blk.attn
        specs[attn.to_q.weight] = col
        specs[attn.to_k.weight] = col
        specs[attn.to_v.weight] = col
        specs[attn.to_out[0].weight] = row
        specs[attn.add_q_proj.weight] = col
        specs[attn.add_k_proj.weight] = col
        specs[attn.add_v_proj.weight] = col
        specs[attn.to_add_out.weight] = row
        # FeedForward: net[0] is GELU(proj), net[2] is the output Linear.
        specs[blk.ff.net[0].proj.weight] = col
        specs[blk.ff.net[2].weight] = row
        specs[blk.ff_context.net[0].proj.weight] = col
        specs[blk.ff_context.net[2].weight] = row

    # Single-stream blocks: pre_only attention (no to_out) + fused mlp/proj_out.
    for blk in model.single_transformer_blocks:
        attn = blk.attn
        specs[attn.to_q.weight] = col
        specs[attn.to_k.weight] = col
        specs[attn.to_v.weight] = col
        specs[blk.proj_mlp.weight] = col
        # proj_out is split into attn/mlp halves (see apply_single_block_rewrite)
        # to avoid the wide concat; both halves are row-parallel (their input
        # feature dim is sharded along model_axis -> one all-reduce on the sum).
        specs[blk.proj_out_attn.weight] = row
        specs[blk.proj_out_mlp.weight] = row

    return specs
