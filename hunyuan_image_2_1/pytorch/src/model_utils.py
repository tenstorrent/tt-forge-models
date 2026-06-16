# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for HunyuanImage 2.1 (Distilled).

Model: hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers
Components:
  - text_encoder:   Qwen2.5-VL-7B-Instruct encoder (8.29B)
  - text_encoder_2: ByT5 encoder                   (0.22B)
  - transformer:    HunyuanImageTransformer2DModel (MMDiT) (17.45B)
  - vae:            AutoencoderKLHunyuanImage      (0.41B)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants (2048 x 2048, single image)
# ---------------------------------------------------------------------------

IMAGE_H = 2048
IMAGE_W = 2048
VAE_SCALE_FACTOR = 32  # AutoencoderKLHunyuanImage.spatial_compression_ratio
LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 64
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 64

NUM_CHANNELS_LATENTS = 64  # transformer.config.in_channels
TRANSFORMER_IN_CHANNELS = NUM_CHANNELS_LATENTS  # no extra conditioning concat in T2I

# Text encoder hidden dims
TEXT_EMBED_DIM = 3584  # Qwen2.5-VL hidden_state dim
TEXT_EMBED_2_DIM = 1472  # ByT5 hidden_state dim

# Tokenizer max lengths driven by the pipeline
TEXT_TOKEN_MAX_LEN = 1034  # tokenizer_max_length (1000) + drop_idx (34)
TEXT_TOKEN_2_MAX_LEN = 128

# Sequence lengths landing in the transformer (after pipeline slicing)
TRANSFORMER_TEXT_SEQ = 1000  # encoder_hidden_states seq dim (post drop_idx)
TRANSFORMER_TEXT_2_SEQ = 128  # encoder_hidden_states_2 seq dim

# Vocabulary sizes
QWEN_VOCAB_SIZE = 151936  # Qwen2.5-VL text encoder
BYT5_VOCAB_SIZE = 384  # ByT5 text_encoder_2

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load Qwen2.5-VL text encoder from the text_encoder subfolder.

    The pipeline only feeds input_ids/attention_mask (no pixel_values), so the
    vision tower never runs. Return just `.language_model` to avoid uploading
    the unused ~0.68B-param visual tower (replicated on every chip).
    """
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()
    return getattr(encoder, "language_model", encoder)


def load_text_encoder_2(dtype: torch.dtype = DTYPE):
    """Load ByT5 text encoder from the text_encoder_2 subfolder."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load HunyuanImageTransformer2DModel from the transformer subfolder."""
    from diffusers import HunyuanImageTransformer2DModel

    return HunyuanImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLHunyuanImage from the vae subfolder."""
    from diffusers import AutoencoderKLHunyuanImage

    return AutoencoderKLHunyuanImage.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — keep forward signatures tensor-in / tensor-out for the
# tt-xla graph runner.
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLHunyuanImage decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class HunyuanImage21TransformerWrapper(torch.nn.Module):
    """Simplify HunyuanImageTransformer2DModel forward to tensor-only inputs/outputs."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        guidance,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for Qwen2.5-VL text encoder.

    Column-parallel (q, k, v, gate, up): ("model", "batch")
    Row-parallel   (o, down):            ("batch", "model")
    """
    specs = {}

    # Unwrap Qwen2_5_VLModel -> decoder; else embed_tokens/layers aren't found
    # and every weight stays replicated on each device -> DRAM OOM.
    encoder = getattr(encoder, "language_model", encoder)

    if hasattr(encoder, "embed_tokens"):
        specs[encoder.embed_tokens.weight] = (None, "batch")

    layers = getattr(encoder, "layers", None)
    if not layers:
        raise ValueError(
            f"No decoder layers on {type(encoder).__name__}; refusing to run "
            "fully replicated (expected `.layers` after unwrapping)."
        )

    for layer in layers:
        sa = layer.self_attn
        specs[sa.q_proj.weight] = ("model", "batch")
        if sa.q_proj.bias is not None:
            specs[sa.q_proj.bias] = ("model",)
        specs[sa.k_proj.weight] = ("model", "batch")
        if sa.k_proj.bias is not None:
            specs[sa.k_proj.bias] = ("model",)
        specs[sa.v_proj.weight] = ("model", "batch")
        if sa.v_proj.bias is not None:
            specs[sa.v_proj.bias] = ("model",)
        specs[sa.o_proj.weight] = ("batch", "model")

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)

    if hasattr(encoder, "norm"):
        specs[encoder.norm.weight] = ("batch",)

    return specs


# text_encoder_2 (ByT5, 0.22B params) fits on a single chip — no sharding.


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for HunyuanImageTransformer2DModel.

    HunyuanImage uses a hybrid MMDiT design with two block lists:
      - transformer_blocks: dual-stream (image + text streams with cross attention)
      - single_transformer_blocks: single-stream (concatenated tokens)

    Column-parallel (Q, K, V, FFN up): ("model", "batch")
    Row-parallel   (O, FFN down):      ("batch", "model")
    """
    specs = {}

    if hasattr(transformer, "x_embedder") and hasattr(transformer.x_embedder, "proj"):
        # patch-embed conv weight: (out_C, in_C, kH, kW)
        specs[transformer.x_embedder.proj.weight] = ("batch", None, None, None)
        if transformer.x_embedder.proj.bias is not None:
            specs[transformer.x_embedder.proj.bias] = ("batch",)

    def _shard_attn(attn):
        for proj_name in (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ):
            if hasattr(attn, proj_name) and getattr(attn, proj_name) is not None:
                proj = getattr(attn, proj_name)
                specs[proj.weight] = ("model", "batch")
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)
        for proj_name in ("to_out", "to_add_out"):
            if hasattr(attn, proj_name) and getattr(attn, proj_name) is not None:
                out = getattr(attn, proj_name)
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

    def _shard_ff(ff):
        # diffusers FeedForward = Sequential(GEGLU, Dropout, Linear)
        if hasattr(ff, "net"):
            if hasattr(ff.net[0], "proj"):
                specs[ff.net[0].proj.weight] = ("model", "batch")
                if ff.net[0].proj.bias is not None:
                    specs[ff.net[0].proj.bias] = ("model",)
            specs[ff.net[2].weight] = ("batch", "model")
            if ff.net[2].bias is not None:
                specs[ff.net[2].bias] = ("batch",)

    # Dual-stream MMDiT blocks
    for block in getattr(transformer, "transformer_blocks", []):
        for norm_name in ("norm1", "norm1_context"):
            if hasattr(block, norm_name) and hasattr(
                getattr(block, norm_name), "linear"
            ):
                lin = getattr(block, norm_name).linear
                specs[lin.weight] = ("model", "batch")
                if lin.bias is not None:
                    specs[lin.bias] = ("model",)
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        for ff_name in ("ff", "ff_context"):
            if hasattr(block, ff_name):
                _shard_ff(getattr(block, ff_name))

    # Single-stream blocks (fused QKV + FFN in one projection in some variants)
    for block in getattr(transformer, "single_transformer_blocks", []):
        if hasattr(block, "norm") and hasattr(block.norm, "linear"):
            lin = block.norm.linear
            specs[lin.weight] = ("model", "batch")
            if lin.bias is not None:
                specs[lin.bias] = ("model",)
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        # single block typically has its own proj_out after concat(attn || mlp)
        if hasattr(block, "proj_out"):
            specs[block.proj_out.weight] = ("batch", "model")
            if block.proj_out.bias is not None:
                specs[block.proj_out.bias] = ("batch",)

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
