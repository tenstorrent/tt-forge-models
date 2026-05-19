# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for HunyuanVideo.

Model: hunyuanvideo-community/HunyuanVideo
Components:
  - text_encoder:   LLaMA-3 encoder
  - text_encoder_2: CLIP text encoder
  - transformer:    HunyuanVideoTransformer3DModel DiT
  - vae:            AutoencoderKLHunyuanVideo
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "hunyuanvideo-community/HunyuanVideo"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
#
# VAE: temporal_compression=4, spatial_compression=8.
#   latent_frames = (5 - 1) // 4 + 1 = 2
#   latent_h/w    = 128 // 8 = 16
# ---------------------------------------------------------------------------

NUM_FRAMES = 5
NUM_LATENT_FRAMES = 2
LATENT_H = 16
LATENT_W = 16

NUM_CHANNELS_LATENTS = 16  # VAE latent channels
TRANSFORMER_IN_CHANNELS = 16  # transformer in_channels (= latent channels)

TEXT_EMBED_DIM = 4096  # LLaMA-3 hidden_state dim
TEXT_EMBED_2_DIM = 768  # CLIP text_encoder pooled projection dim

TEXT_TOKEN_MAX_LEN = 256  # LLaMA tokenized prompt length
TEXT_TOKEN_2_MAX_LEN = 77  # CLIP tokenizer max length
TRANSFORMER_TEXT_SEQ = 256  # encoder_hidden_states seq dim into transformer

# Vocabulary sizes
LLAMA_VOCAB_SIZE = 128256  # LLaMA-3 text encoder
CLIP_VOCAB_SIZE = 49408  # CLIP text_encoder_2

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load LLaMA text encoder from the text_encoder subfolder."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_text_encoder_2(dtype: torch.dtype = DTYPE):
    """Load CLIP text encoder from the text_encoder_2 subfolder."""
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load HunyuanVideoTransformer3DModel from the transformer subfolder."""
    from diffusers import HunyuanVideoTransformer3DModel

    return HunyuanVideoTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLHunyuanVideo from the vae subfolder."""
    from diffusers import AutoencoderKLHunyuanVideo

    return AutoencoderKLHunyuanVideo.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for the LLaMA text encoder: [input_ids, attention_mask]."""
    input_ids = torch.randint(
        0, LLAMA_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
    )
    attention_mask = torch.ones(1, TEXT_TOKEN_MAX_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_text_encoder_2_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for the CLIP text encoder: [input_ids, attention_mask]."""
    input_ids = torch.randint(
        0, CLIP_VOCAB_SIZE, (1, TEXT_TOKEN_2_MAX_LEN), dtype=torch.long
    )
    attention_mask = torch.ones(1, TEXT_TOKEN_2_MAX_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for HunyuanVideoTransformer3DModel.

    Returns [hidden_states, timestep, encoder_hidden_states,
             encoder_attention_mask, pooled_projections].
    """
    hidden_states = torch.randn(
        1,
        TRANSFORMER_IN_CHANNELS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    timestep = torch.tensor([1000.0], dtype=dtype)
    encoder_hidden_states = torch.randn(
        1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM, dtype=dtype
    )
    encoder_attention_mask = torch.ones(1, TRANSFORMER_TEXT_SEQ, dtype=dtype)
    pooled_projections = torch.randn(1, TEXT_EMBED_2_DIM, dtype=dtype)
    return [
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
    ]


def load_vae_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic latent input for VAEDecoderWrapper: [z (1,16,2,16,16)]."""
    z = torch.randn(
        1,
        NUM_CHANNELS_LATENTS,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    return [z]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLHunyuanVideo decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class HunyuanVideoTransformerWrapper(torch.nn.Module):
    """Simplify HunyuanVideoTransformer3DModel forward to tensor-only inputs/outputs."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for LLaMA text encoder.

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


def shard_text_encoder_2_specs(encoder) -> dict:
    """Shard specs for CLIP text_encoder_2.

    Column-parallel (q, k, v, fc1): ("model", "batch")
    Row-parallel   (out_proj, fc2): ("batch", "model")
    """
    specs = {}

    text_model = getattr(encoder, "text_model", encoder)

    if hasattr(text_model, "embeddings"):
        emb = text_model.embeddings
        if hasattr(emb, "token_embedding"):
            specs[emb.token_embedding.weight] = (None, "batch")
        if hasattr(emb, "position_embedding"):
            specs[emb.position_embedding.weight] = (None, "batch")

    layers = getattr(getattr(text_model, "encoder", None), "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name)
            specs[proj.weight] = ("model", "batch")
            if proj.bias is not None:
                specs[proj.bias] = ("model",)
        specs[sa.out_proj.weight] = ("batch", "model")
        if sa.out_proj.bias is not None:
            specs[sa.out_proj.bias] = ("batch",)

        mlp = layer.mlp
        specs[mlp.fc1.weight] = ("model", "batch")
        if mlp.fc1.bias is not None:
            specs[mlp.fc1.bias] = ("model",)
        specs[mlp.fc2.weight] = ("batch", "model")
        if mlp.fc2.bias is not None:
            specs[mlp.fc2.bias] = ("batch",)

        specs[layer.layer_norm1.weight] = ("batch",)
        if layer.layer_norm1.bias is not None:
            specs[layer.layer_norm1.bias] = ("batch",)
        specs[layer.layer_norm2.weight] = ("batch",)
        if layer.layer_norm2.bias is not None:
            specs[layer.layer_norm2.bias] = ("batch",)

    if hasattr(text_model, "final_layer_norm"):
        specs[text_model.final_layer_norm.weight] = ("batch",)
        if text_model.final_layer_norm.bias is not None:
            specs[text_model.final_layer_norm.bias] = ("batch",)

    return specs


def _causal_conv3d_weight(conv_module):
    """Resolve weight/bias tensors for a HunyuanVideoCausalConv3d (or plain Conv3d)."""
    inner = getattr(conv_module, "conv", conv_module)
    return inner.weight, getattr(inner, "bias", None)


def _shard_resnet_block(block, specs: dict) -> None:
    """Shard specs for a decoder ResNet block.

    Megatron-style pair (conv1 col-parallel, conv2 row-parallel) with a
    replicated shortcut so the residual add works without re-sharding:

      conv1 (col-parallel):
        weight ("model", None, None, None, None)   # shard Cout
        bias   ("model",)                          # matches Cout

      conv2 (row-parallel):
        weight (None, "model", None, None, None)   # shard Cin
        bias   (None,)                             # output is replicated after
                                                   # the all_reduce, so bias
                                                   # cannot be sharded
                                                   # (would otherwise be summed
                                                   #  N times by the reduce)

      conv_shortcut (replicated):
        weight (None, None, None, None, None)
        bias   (None,)
        # conv2 produces a replicated output; the residual add
        # `conv2_out + shortcut_out` requires the shortcut to also be
        # replicated, so we don't shard it.
    """
    if hasattr(block, "norm1"):
        specs[block.norm1.weight] = ("batch",)
        if block.norm1.bias is not None:
            specs[block.norm1.bias] = ("batch",)

    w1, b1 = _causal_conv3d_weight(block.conv1)
    specs[w1] = ("model", None, None, None, None)
    if b1 is not None:
        specs[b1] = ("model",)

    if hasattr(block, "norm2"):
        specs[block.norm2.weight] = ("batch",)
        if block.norm2.bias is not None:
            specs[block.norm2.bias] = ("batch",)

    w2, b2 = _causal_conv3d_weight(block.conv2)
    specs[w2] = (None, "model", None, None, None)
    if b2 is not None:
        specs[b2] = (None,)

    if getattr(block, "conv_shortcut", None) is not None:
        ws, bs = _causal_conv3d_weight(block.conv_shortcut)
        specs[ws] = (None, None, None, None, None)
        if bs is not None:
            specs[bs] = (None,)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKLHunyuanVideo (decoder-only path).

    For Conv3d weights (out_ch, in_ch, T, H, W) only one channel dim is sharded
    (along "model"), since Conv3d currently supports single-axis channel
    sharding:
      Column-parallel (entry conv, first conv in resnet, upsampler conv):
        weight ("model", None, None, None, None), bias ("model",)
      Row-parallel   (second conv in resnet, exit conv):
        weight (None, "model", None, None, None), bias (None,)
        # bias replicated because row-parallel output is replicated after
        # the all_reduce — a sharded bias would be wrong (summed N times).
      Replicated (conv_shortcut): all dims None for both weight and bias.
        # must match the post-all_reduce replicated state of conv2 so the
        # residual add doesn't need a re-shard.
    Norm / attention specs mirror the LLaMA / transformer conventions.
    """
    specs = {}

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        return specs

    if hasattr(decoder, "conv_in"):
        w, b = _causal_conv3d_weight(decoder.conv_in)
        specs[w] = ("model", None, None, None, None)
        if b is not None:
            specs[b] = ("model",)

    mid_block = getattr(decoder, "mid_block", None)
    if mid_block is not None:
        for resnet in getattr(mid_block, "resnets", []) or []:
            _shard_resnet_block(resnet, specs)
        for attn in getattr(mid_block, "attentions", []) or []:
            if attn is None:
                continue
            if hasattr(attn, "group_norm") and attn.group_norm is not None:
                specs[attn.group_norm.weight] = ("batch",)
                if attn.group_norm.bias is not None:
                    specs[attn.group_norm.bias] = ("batch",)
            for proj_name in ("to_q", "to_k", "to_v"):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", "batch")
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            if hasattr(attn, "to_out"):
                out = attn.to_out
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

    for up_block in getattr(decoder, "up_blocks", []) or []:
        for resnet in getattr(up_block, "resnets", []) or []:
            _shard_resnet_block(resnet, specs)
        for upsampler in getattr(up_block, "upsamplers", []) or []:
            w, b = _causal_conv3d_weight(upsampler.conv)
            specs[w] = ("model", None, None, None, None)
            if b is not None:
                specs[b] = ("model",)

    if hasattr(decoder, "conv_norm_out"):
        specs[decoder.conv_norm_out.weight] = ("batch",)
        if decoder.conv_norm_out.bias is not None:
            specs[decoder.conv_norm_out.bias] = ("batch",)

    if hasattr(decoder, "conv_out"):
        w, b = _causal_conv3d_weight(decoder.conv_out)
        specs[w] = (None, "model", None, None, None)
        if b is not None:
            specs[b] = (None,)

    return specs


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for HunyuanVideoTransformer3DModel.

    Column-parallel (Q, K, V, FFN up): ("model", "batch")
    Row-parallel   (O, FFN down):      ("batch", "model")
    """
    specs = {}

    if hasattr(transformer, "x_embedder") and hasattr(transformer.x_embedder, "proj"):
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

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
