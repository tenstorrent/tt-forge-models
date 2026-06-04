# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for FLUX.2-dev per-component loading.

FLUX.2-dev (black-forest-labs/FLUX.2-dev) is a text-to-image DiffusionPipeline
(``Flux2Pipeline``) with three compilable components:

- ``text_encoder``: Mistral3ForConditionalGeneration (~24.0B params, bf16)
- ``transformer``:  Flux2Transformer2DModel        (~32.2B params, bf16)
- ``vae``:          AutoencoderKLFlux2             (~84M params, fp32)

All shapes/dtypes below were captured from one real CPU forward pass at
64x64 / 2 denoise steps (max_sequence_length=512). See
``.claude/bringup/flux_2_dev/io_spec_raw.jsonl`` for the raw capture.

The pipeline packs latents into ``(H/16)*(W/16)`` tokens of 128 channels
(latent_channels=32, vae_scale_factor=8, patch=2 -> 32*4=128). RoPE is 4D
(``axes_dims_rope=[32,32,32,32]``), hence ``img_ids``/``txt_ids`` have a
trailing dim of 4.
"""

from typing import Any

import torch

# --------------------------------------------------------------------------
# Captured I/O spec (64x64, seq=512). Reproduced here so the loader is
# self-contained without re-running capture.
# --------------------------------------------------------------------------

VAE_SCALE_FACTOR = 8
PATCH = 2
LATENT_CHANNELS = 32
TRANSFORMER_IN_CHANNELS = 128  # latent_channels * patch^2
JOINT_ATTENTION_DIM = 15360  # encoder_hidden_states feature dim
TEXT_ENCODER_HIDDEN = 5120
ROPE_AXES = 4  # len(axes_dims_rope)

# Capture defaults (tiny, fast CPU golden for the single_device VAE test).
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_SEQ_LEN = 512


# --------------------------------------------------------------------------
# Component loaders -- each returns ONLY the requested component, freeing the
# rest of the pipeline so we never hold all ~56B params in RAM at once.
# --------------------------------------------------------------------------


def _load_component(pretrained_model_name: str, attr: str, dtype: torch.dtype):
    from diffusers import Flux2Pipeline

    pipe = Flux2Pipeline.from_pretrained(pretrained_model_name, torch_dtype=dtype)
    component = getattr(pipe, attr)
    # Drop references to the other components before returning.
    del pipe
    component.eval()
    return component


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load Flux2Transformer2DModel (~32.2B params) only."""
    from diffusers import Flux2Transformer2DModel

    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name, subfolder="transformer", torch_dtype=dtype
    )
    transformer.eval()
    return transformer


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKLFlux2 (~84M params) only."""
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name, subfolder="vae", torch_dtype=dtype
    )
    vae.eval()
    return vae


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load Mistral3ForConditionalGeneration text encoder (~24.0B params) only."""
    from transformers import Mistral3ForConditionalGeneration

    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        pretrained_model_name, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.eval()
    return text_encoder


class VAEDecoderWrapper(torch.nn.Module):
    """Run AutoencoderKLFlux2.decode and return the reconstructed pixel tensor.

    Captured: latent [1, 32, H/8, W/8] -> sample [1, 3, H, W].
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# --------------------------------------------------------------------------
# Synthetic input builders -- reproduce the captured shapes/dtypes. Inputs
# scale with image height/width and text sequence length using the pipeline's
# packing formulas; defaults reproduce the exact captured 64x64 / seq=512 pass.
# Do NOT call the pipeline here -- component isolation requires synthetic input.
# --------------------------------------------------------------------------


def _tokens(height: int, width: int) -> int:
    return (height // (VAE_SCALE_FACTOR * PATCH)) * (
        width // (VAE_SCALE_FACTOR * PATCH)
    )


def load_transformer_inputs(
    dtype: torch.dtype = torch.bfloat16,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> dict:
    """Inputs for Flux2Transformer2DModel.forward (captured 64x64/seq=512)."""
    tokens = _tokens(height, width)
    return {
        "hidden_states": torch.randn(1, tokens, TRANSFORMER_IN_CHANNELS, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            1, seq_len, JOINT_ATTENTION_DIM, dtype=dtype
        ),
        "timestep": torch.tensor([1.0], dtype=dtype),
        "img_ids": torch.zeros(1, tokens, ROPE_AXES, dtype=torch.int64),
        "txt_ids": torch.zeros(1, seq_len, ROPE_AXES, dtype=torch.int64),
        "guidance": torch.tensor([4.0], dtype=torch.float32),
    }


def load_vae_decoder_inputs(
    dtype: torch.dtype = torch.bfloat16,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> torch.Tensor:
    """Latent input for AutoencoderKLFlux2.decode (captured [1,32,8,8] at 64x64)."""
    return torch.randn(
        1,
        LATENT_CHANNELS,
        height // VAE_SCALE_FACTOR,
        width // VAE_SCALE_FACTOR,
        dtype=dtype,
    )


def load_text_encoder_inputs(
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = 131072,
) -> dict:
    """Inputs for Mistral3ForConditionalGeneration.forward (captured seq=512).

    ``dtype`` is accepted for API symmetry; the integer inputs ignore it.

    ``logits_to_keep=1`` makes lm_head run on only the last token, so the
    output ``logits`` is [1, 1, 131072] instead of [1, 512, 131072] (~134 MB
    replicated bf16). The FLUX.2 pipeline consumes ``hidden_states`` (collected
    from every decoder layer regardless of logits_to_keep), never the logits,
    so this drops the dominant activation buffer without changing what we test.
    """
    return {
        "input_ids": torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.int64),
        "output_hidden_states": True,
        "logits_to_keep": 1,
    }


# --------------------------------------------------------------------------
# SPMD mesh + Megatron-1D shard specs (PROMOTION-ONLY).
#
# These are best-effort scaffolds derived from the meta-device module tree;
# they are validated/refined by /model-bringup-multichip and
# /model-bringup-repair-shard-spec before any multichip run. Single-chip
# bringup does NOT use them (the VAE is single_device; the transformer and
# text_encoder are weight-bound on every single chip -> promotion-only).
#
# Mesh axes: (batch, model). "model" is the tensor-parallel shard axis;
# "batch" stays data-parallel/replicated. Column-parallel weights -> ("model",
# None); row-parallel weights -> (None, "model") with replicated bias (None,).
# --------------------------------------------------------------------------

MESH_NAMES = (None, "model")
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 32: (8, 4)}


def get_mesh_config(num_devices: int):
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count: {num_devices}. "
            f"Expected one of {sorted(MESH_SHAPES)}."
        )
    return MESH_SHAPES[num_devices], MESH_NAMES


def shard_transformer_specs(transformer) -> dict:
    """Megatron-1D specs for Flux2Transformer2DModel.

    Double-stream blocks (``transformer_blocks``): attention to_q/to_k/to_v and
    text add_q/add_k/add_v are column-parallel; to_out[0]/to_add_out are
    row-parallel; SwiGLU FF linear_in is column-parallel, linear_out is
    row-parallel (both image ``ff`` and text ``ff_context``).

    Single-stream blocks (``single_transformer_blocks``): the fused
    ``to_qkv_mlp_proj`` is column-parallel and ``to_out`` is row-parallel.

    Embedders, modulation, norms and proj_out are left replicated.
    """
    specs: dict = {}

    def col(linear):
        if linear is None:
            return
        specs[linear.weight] = ("model", None)
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("model",)

    def row(linear):
        if linear is None:
            return
        specs[linear.weight] = (None, "model")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = (None,)  # replicated (added after all-reduce)

    for block in transformer.transformer_blocks:
        attn = block.attn
        col(attn.to_q)
        col(attn.to_k)
        col(attn.to_v)
        col(getattr(attn, "add_q_proj", None))
        col(getattr(attn, "add_k_proj", None))
        col(getattr(attn, "add_v_proj", None))
        row(attn.to_out[0])
        row(getattr(attn, "to_add_out", None))
        col(block.ff.linear_in)
        row(block.ff.linear_out)
        if getattr(block, "ff_context", None) is not None:
            col(block.ff_context.linear_in)
            row(block.ff_context.linear_out)

    for block in transformer.single_transformer_blocks:
        col(block.attn.to_qkv_mlp_proj)
        row(block.attn.to_out)

    # Conditioning / embedding projections — row-parallel (shard contraction /
    # in_features dim). Each takes a REPLICATED input and must produce a
    # REPLICATED output (LayerNorm / residual / AdaLN modulation operate on the
    # full hidden dim), so column-parallel is wrong here; row-parallel shards
    # the weight + the large pre-all-reduce intermediate, then all-reduces back
    # to the full replicated activation (no activation growth).
    #
    # Added at REPAIR after iter2 DRAM OOM: these were ~1.63 GB/chip of
    # replicated weight (modulation linears dominate) crowding activation space
    # on the maxed 8-chip mesh. Sharding them frees ~1.29 GB/chip.
    # in_features all divisible by 8: context_embedder=15360, modulation/
    # norm_out=6144. x_embedder (128->6144) and proj_out (6144->128) are tiny
    # (<1M) and left replicated.
    row(transformer.context_embedder)
    row(transformer.norm_out.linear)
    row(transformer.double_stream_modulation_img.linear)
    row(transformer.double_stream_modulation_txt.linear)
    row(transformer.single_stream_modulation.linear)

    return specs


def shard_text_encoder_specs(text_encoder) -> dict:
    """Megatron-1D specs for the Mistral3 language-model decoder layers.

    Per layer: self_attn q/k/v_proj column-parallel, o_proj row-parallel;
    mlp gate_proj/up_proj column-parallel, down_proj row-parallel. Embeddings,
    norms, lm_head and the vision tower are left replicated.
    """
    specs: dict = {}
    layers = text_encoder.model.language_model.layers
    for layer in layers:
        attn = layer.self_attn
        specs[attn.q_proj.weight] = ("model", None)
        specs[attn.k_proj.weight] = ("model", None)
        specs[attn.v_proj.weight] = ("model", None)
        specs[attn.o_proj.weight] = (None, "model")
        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", None)
        specs[mlp.up_proj.weight] = ("model", None)
        specs[mlp.down_proj.weight] = (None, "model")
    return specs
