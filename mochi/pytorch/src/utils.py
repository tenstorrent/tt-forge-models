# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Mochi model loading."""

from typing import Any

import torch

# Channel-wise standard deviations for VAE latent normalization
# Source: Mochi VAE implementation (12 latent channels)
VAE_STD_CHANNELS = [
    0.925,
    0.934,
    0.946,
    0.939,
    0.961,
    1.033,
    0.979,
    1.024,
    0.983,
    1.046,
    0.964,
    1.004,
]


LATENT_HEIGHT = 12
LATENT_WIDTH = 12
LATENT_DEPTH = 2


def normalize_latents(latent: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """
    Normalize VAE latents with channel-wise standard deviations.

    Mochi VAE expects normalized latents as input. Each of the 12 latent
    channels has a specific standard deviation value.

    Args:
        latent: Input latent tensor of shape [B, 12, t, h, w]
        device: Target device for normalization tensor (defaults to latent.device)
        dtype: Target dtype for normalization tensor (defaults to latent.dtype)

    Returns:
        Normalized latent tensor of same shape as input
    """
    if device is None:
        device = latent.device
    if dtype is None:
        dtype = latent.dtype

    vae_std = torch.tensor(VAE_STD_CHANNELS, dtype=dtype, device=device)
    # Reshape to [1, 12, 1, 1, 1] for broadcasting
    vae_std = vae_std.view(1, 12, 1, 1, 1)

    return latent / vae_std


# ============================================================================
# Model Loading Functions
# ============================================================================


def load_pipeline(
    pretrained_model_name: str,
    dtype: torch.dtype,
    enable_tiling: bool = False,
    tile_sample_min_height: int = 128,
    tile_sample_min_width: int = 128,
    tile_sample_stride_height: int = 128,
    tile_sample_stride_width: int = 128,
):
    """
    Load the full MochiPipeline.

    Components loaded:
    - scheduler: FlowMatchEulerDiscreteScheduler
    - vae: AutoencoderKLMochi
    - text_encoder: T5EncoderModel
    - tokenizer: T5TokenizerFast
    - transformer: MochiTransformer3DModel
    """
    from diffusers import MochiPipeline

    pipeline = MochiPipeline.from_pretrained(pretrained_model_name, torch_dtype=dtype)

    if enable_tiling:
        pipeline.vae.enable_tiling(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
        )
        pipeline.vae.drop_last_temporal_frames = False

    pipeline.eval()

    return pipeline


def load_vae(
    pretrained_model_name: str,
    dtype: torch.dtype,
    enable_tiling: bool = False,
    tile_sample_min_height: int = 128,
    tile_sample_min_width: int = 128,
    tile_sample_stride_height: int = 128,
    tile_sample_stride_width: int = 128,
):

    from diffusers import AutoencoderKLMochi

    vae = AutoencoderKLMochi.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    )

    if enable_tiling:
        vae.enable_tiling(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
        )
        vae.drop_last_temporal_frames = False

    vae.eval()

    return vae


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load MochiTransformer3DModel (~10B params).

    The transformer has:

    Forward signature:
        forward(hidden_states, encoder_hidden_states, timestep,
               encoder_attention_mask, attention_kwargs=None, return_dict=True)

    Where:
    - hidden_states: (batch, num_channels, num_frames, height, width)
    - encoder_hidden_states: text embeddings from T5
    - timestep: diffusion timestep
    - encoder_attention_mask: attention mask for text
    """
    from diffusers import MochiTransformer3DModel

    transformer = MochiTransformer3DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    transformer.eval()

    return transformer


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load T5EncoderModel (T5-XXL variant).

    The text encoder converts text prompts into embeddings.
    Output dimension: 4096 (matches transformer's text_embed_dim)
    """
    from transformers import T5EncoderModel

    text_encoder = T5EncoderModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )

    text_encoder.eval()
    return text_encoder


# ============================================================================
# Input Loading Functions
# ============================================================================


def load_vae_decoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Load inputs for VAE decoder.

    Args:
        dtype: Data type for the tensor (default: torch.bfloat16)

    Returns:
        Normalized latent tensor of shape [1, 12, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]
    """
    # [batch, channels, time, height, width]
    latent = torch.randn(1, 12, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype)
    return normalize_latents(latent, dtype=dtype)


def load_vae_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Load inputs for VAE encoder.

    Args:
        dtype: Data type for the tensor (default: torch.bfloat16)

    Returns:
        RGB video tensor of shape [1, 3, LATENT_DEPTH * 6, LATENT_HEIGHT * 8, LATENT_WIDTH * 8]
        (batch, channels, frames, height, width)

    Note:
        The encoder compresses:
        - Temporal: 6x (12 frames -> 2 latent frames)
        - Spatial: 8x8 (96x96 -> 12x12)
    """
    # [batch, channels, frames, height, width]
    # Using small test dimensions that match decoder output expectations
    return torch.randn(
        1, 3, LATENT_DEPTH * 6, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype
    )


def load_transformer_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Load inputs for transformer.

    Args:
        dtype: Data type for tensors (default: torch.bfloat16)

    Returns:
        Dict with:
        - hidden_states: (batch, channels, frames, height, width)
        - encoder_hidden_states: (batch, seq_len, embed_dim)
        - timestep: (batch,)
        - encoder_attention_mask: (batch, seq_len)
    """
    batch_size = 1
    num_channels = 12  # in_channels
    num_frames = LATENT_DEPTH
    height = LATENT_HEIGHT
    width = LATENT_WIDTH
    seq_len = 128  # max_sequence_length
    text_embed_dim = 4096

    return {
        "hidden_states": torch.randn(
            batch_size, num_channels, num_frames, height, width, dtype=dtype
        ),
        "encoder_hidden_states": torch.randn(
            batch_size, seq_len, text_embed_dim, dtype=dtype
        ),
        "timestep": torch.tensor([500], dtype=torch.long),
        "encoder_attention_mask": torch.ones(batch_size, seq_len, dtype=dtype),
    }


def load_text_encoder_inputs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """
    Load inputs for text encoder.

    Args:
        dtype: Data type (unused, but kept for API consistency)

    Returns:
        Dict with input_ids and attention_mask
    """
    batch_size = 1
    seq_len = 32

    return {
        "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


# ============================================================================
# Original-resolution input helpers
# ============================================================================

ORIGINAL_LATENT_HEIGHT = 60  # 480 / 8
ORIGINAL_LATENT_WIDTH = 106  # 848 / 8
ORIGINAL_LATENT_DEPTH = 14  # (num_frames=84 - 1) // 6 + 1
ORIGINAL_MAX_SEQ_LEN = 256
ORIGINAL_TEXT_EMBED_DIM = 4096
ORIGINAL_LATENT_CHANNELS = 12
ORIGINAL_T5_VOCAB_SIZE = 32128


def load_transformer_inputs_full_res(
    dtype: torch.dtype = torch.bfloat16,
) -> list:
    """Original-resolution positional inputs for MochiTransformer3DModel."""
    hidden_states = torch.randn(
        1,
        ORIGINAL_LATENT_CHANNELS,
        ORIGINAL_LATENT_DEPTH,
        ORIGINAL_LATENT_HEIGHT,
        ORIGINAL_LATENT_WIDTH,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        1, ORIGINAL_MAX_SEQ_LEN, ORIGINAL_TEXT_EMBED_DIM, dtype=dtype
    )
    timestep = torch.tensor([500], dtype=dtype)
    encoder_attention_mask = torch.ones(1, ORIGINAL_MAX_SEQ_LEN, dtype=torch.bool)
    return [hidden_states, encoder_hidden_states, timestep, encoder_attention_mask]


def load_vae_decoder_inputs_full_res(
    dtype: torch.dtype = torch.bfloat16,
) -> list:
    """Original-resolution latent for the Mochi VAE decoder."""
    latent = torch.randn(
        1,
        ORIGINAL_LATENT_CHANNELS,
        ORIGINAL_LATENT_DEPTH,
        ORIGINAL_LATENT_HEIGHT,
        ORIGINAL_LATENT_WIDTH,
        dtype=dtype,
    )
    return [latent]


def load_text_encoder_inputs_full_res(
    dtype: torch.dtype = torch.bfloat16,
) -> list:
    """Original-resolution positional inputs for T5-XXL text encoder."""
    input_ids = torch.randint(
        0, ORIGINAL_T5_VOCAB_SIZE, (1, ORIGINAL_MAX_SEQ_LEN), dtype=torch.long
    )
    attention_mask = torch.ones(1, ORIGINAL_MAX_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


# ============================================================================
# SPMD shard specifications (transformer)
# ============================================================================

# (batch, model) mesh shapes by device count.
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 32: (8, 4)}
MESH_NAMES = (None, "model")


def shard_transformer_specs(transformer) -> dict:
    """Build tensor -> partition_spec dict for MochiTransformer3DModel.

    Mesh axes: ("batch", "model"). Only "model" is used as a real shard axis
    here; "batch" is the data-parallel axis and stays replicated, so any dim
    that shouldn't be sharded is marked None rather than "batch".

    Column-parallel (Q/K/V, FF up):  ("model", None)
    Row-parallel    (out, FF down):  (None, "model"), bias replicated (None,)
    patch_embed / proj_out / norms:  replicated (not in specs).
    """
    specs: dict = {}

    # patch_embed.proj and proj_out run replicated.

    # Per-block sharding.
    for block in transformer.transformer_blocks:
        attn = block.attn1

        # Q/K/V image stream (column-parallel, bias=False in Mochi).
        specs[attn.to_q.weight] = ("model", None)
        specs[attn.to_k.weight] = ("model", None)
        specs[attn.to_v.weight] = ("model", None)

        # Q/K/V text stream (column-parallel, added_proj_bias=False in Mochi).
        if attn.add_q_proj is not None:
            specs[attn.add_q_proj.weight] = ("model", None)
        specs[attn.add_k_proj.weight] = ("model", None)
        specs[attn.add_v_proj.weight] = ("model", None)

        # Output projections (row-parallel). Bias is replicated: the all-reduce
        # after row-parallel matmul produces the full output on every chip, so
        # each chip adds the full bias once.
        specs[attn.to_out[0].weight] = (None, "model")
        if attn.to_out[0].bias is not None:
            specs[attn.to_out[0].bias] = (None,)
        if not attn.context_pre_only and hasattr(attn, "to_add_out"):
            specs[attn.to_add_out.weight] = (None, "model")
            if attn.to_add_out.bias is not None:
                specs[attn.to_add_out.bias] = (None,)

        # FeedForward (SwiGLU): net[0].proj is Linear(dim, 2*inner) -> column-parallel,
        # net[2] is Linear(inner, dim) -> row-parallel. Mochi FF bias=False.
        specs[block.ff.net[0].proj.weight] = ("model", None)
        specs[block.ff.net[2].weight] = (None, "model")
        if block.ff_context is not None:
            specs[block.ff_context.net[0].proj.weight] = ("model", None)
            specs[block.ff_context.net[2].weight] = (None, "model")

    return specs


def shard_vae_decoder_specs(decoder) -> dict:
    """Build tensor -> partition_spec dict for MochiDecoder3D.

    Megatron-style pairing on each ResBlock's two convs:
      conv1 (CogVideoXCausalConv3d): column-parallel on out_channels
      conv2 (CogVideoXCausalConv3d): row-parallel on in_channels, bias replicated
      norm2 (MochiChunkedGroupNorm3D): channel-sharded
    norm1, top-level conv_in, proj_out, and per-up-block proj are left replicated.
    """
    specs: dict = {}

    def shard_resnet(resnet):
        # conv1: column-parallel (shard out channels = dim 0 of [O, I, kD, kH, kW])
        specs[resnet.conv1.conv.weight] = ("model", None, None, None, None)
        if resnet.conv1.conv.bias is not None:
            specs[resnet.conv1.conv.bias] = ("model",)

        # GroupNorm between conv1 and conv2 — channel-sharded to match conv1's
        # output sharding. norm1 sees the (replicated) ResBlock input, so its
        # weights stay replicated.
        if resnet.norm2.norm_layer.weight is not None:
            specs[resnet.norm2.norm_layer.weight] = ("model",)
        if resnet.norm2.norm_layer.bias is not None:
            specs[resnet.norm2.norm_layer.bias] = ("model",)

        # conv2: row-parallel; bias is replicated (added after the implicit
        # all-reduce that follows row-parallel matmul).
        specs[resnet.conv2.conv.weight] = (None, "model", None, None, None)
        if resnet.conv2.conv.bias is not None:
            specs[resnet.conv2.conv.bias] = (None,)

    # block_in (MochiMidBlock3D) — resnets only; attentions are None in decoder.
    for resnet in decoder.block_in.resnets:
        shard_resnet(resnet)

    # up_blocks (MochiUpBlock3D) — resnets per block; up_block.proj
    # (unpatchify linear) is left replicated.
    for up_block in decoder.up_blocks:
        for resnet in up_block.resnets:
            shard_resnet(resnet)

    # block_out (MochiMidBlock3D).
    for resnet in decoder.block_out.resnets:
        shard_resnet(resnet)

    return specs
