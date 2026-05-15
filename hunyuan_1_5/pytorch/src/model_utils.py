# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for HunyuanVideo 1.5 (480p t2v distilled).

Model: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled
Components:
  - text_encoder:   Qwen2.5-VL encoder (7.07B)
  - text_encoder_2: ByT5 encoder (0.22B)
  - transformer:    HunyuanVideo15Transformer3DModel DiT (8.33B)
  - vae:            AutoencoderKLHunyuanVideo15 (1.26B)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
# ---------------------------------------------------------------------------

NUM_FRAMES = 17  # smallest valid (4k+1) — for smoke
NUM_LATENT_FRAMES = 5  # (17-1)//4 + 1
LATENT_H = 30  # 480 // 16
LATENT_W = 53  # 848 // 16

NUM_CHANNELS_LATENTS = 32  # VAE latent channels
TRANSFORMER_IN_CHANNELS = 65  # 32 latent + 32 cond + 1 mask

TEXT_EMBED_DIM = 3584  # Qwen2.5-VL hidden_state dim
TEXT_EMBED_2_DIM = 1472  # ByT5 hidden_state dim

IMAGE_EMBED_DIM = 1152  # transformer.config.image_embed_dim
IMAGE_EMBED_SEQ = 64  # pipeline.vision_num_semantic_tokens default

TEXT_TOKEN_MAX_LEN = 1108  # Qwen2.5-VL tokenized prompt length
TEXT_TOKEN_2_MAX_LEN = 256  # ByT5 tokenizer_2_max_length default
TRANSFORMER_TEXT_SEQ = 1000  # encoder_hidden_states seq dim into transformer
TRANSFORMER_TEXT_2_SEQ = 256  # encoder_hidden_states_2 seq dim

# Vocabulary sizes
QWEN_VOCAB_SIZE = 151936  # Qwen2.5-VL text encoder
BYT5_VOCAB_SIZE = 384  # ByT5 text_encoder_2

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load Qwen2.5-VL text encoder from the text_encoder subfolder."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


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
    """Load HunyuanVideo15Transformer3DModel from the transformer subfolder."""
    from diffusers import HunyuanVideo15Transformer3DModel

    return HunyuanVideo15Transformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE, enable_tiling: bool = False):
    """Load AutoencoderKLHunyuanVideo15 from the vae subfolder.

    When enable_tiling=True, the VAE will split spatially-large latents into
    overlapping tiles and blend the outputs — see
    AutoencoderKLHunyuanVideo15.enable_tiling. Lower memory + smaller per-step
    attention seq_len at the cost of tile-boundary approximation.
    """
    from diffusers import AutoencoderKLHunyuanVideo15

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    )
    if enable_tiling:
        vae.enable_tiling()
    return vae.eval()


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLHunyuanVideo15 decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class HunyuanVideo15TransformerWrapper(torch.nn.Module):
    """Simplify HunyuanVideo15Transformer3DModel forward to tensor-only inputs/outputs."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            image_embeds=image_embeds,
            timestep=timestep,
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

    if hasattr(encoder, "embed_tokens"):
        specs[encoder.embed_tokens.weight] = (None, "batch")

    layers = getattr(encoder, "layers", None)
    if layers is None:
        return specs

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


# text_encoder_2 (ByT5, 0.22B params) is small enough to fit on a single chip — no sharding needed.


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for HunyuanVideo15Transformer3DModel.

    Column-parallel (Q, K, V, FFN up): ("model", "batch")
    Row-parallel   (O, FFN down):      ("batch", "model")
    """
    specs = {}

    if hasattr(transformer.x_embedder, "proj"):
        specs[transformer.x_embedder.proj.weight] = ("batch", None, None, None, None)
        if transformer.x_embedder.proj.bias is not None:
            specs[transformer.x_embedder.proj.bias] = ("batch",)

    for block in transformer.transformer_blocks:
        for norm_name in ("norm1", "norm1_context"):
            if hasattr(block, norm_name) and hasattr(
                getattr(block, norm_name), "linear"
            ):
                lin = getattr(block, norm_name).linear
                specs[lin.weight] = ("model", "batch")
                if lin.bias is not None:
                    specs[lin.bias] = ("model",)

        if hasattr(block, "attn"):
            attn = block.attn
            for proj_name in (
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
            ):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", "batch")
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            for proj_name in ("to_out", "to_add_out"):
                if hasattr(attn, proj_name):
                    out = getattr(attn, proj_name)
                    target = (
                        out[0]
                        if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                        else out
                    )
                    specs[target.weight] = ("batch", "model")
                    if target.bias is not None:
                        specs[target.bias] = ("batch",)

        for ff_name in ("ff", "ff_context"):
            if hasattr(block, ff_name):
                ff = getattr(block, ff_name)
                if hasattr(ff.net[0], "proj"):
                    specs[ff.net[0].proj.weight] = ("model", "batch")
                    if ff.net[0].proj.bias is not None:
                        specs[ff.net[0].proj.bias] = ("model",)
                specs[ff.net[2].weight] = ("batch", "model")
                if ff.net[2].bias is not None:
                    specs[ff.net[2].bias] = ("batch",)

    specs[transformer.proj_out.weight] = (None, "batch")
    if transformer.proj_out.bias is not None:
        specs[transformer.proj_out.bias] = (None,)

    return specs


# VAE runs single-device (1.26B params fits); current TT-XLA hits a hang during decode — to debug.
