# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for CogVideoX-5b text-to-video.

Model: THUDM/CogVideoX-5b
Components:
  - text_encoder: T5 v1.1-XXL encoder (~4.76B)
  - transformer:  CogVideoXTransformer3DModel DiT (~5.0B)
  - vae:          AutoencoderKLCogVideoX (~0.22B)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "THUDM/CogVideoX-5b"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants (default 480x720)
# ---------------------------------------------------------------------------

NUM_FRAMES = 1  # video frames at sampling time
NUM_LATENT_FRAMES = 1  # (NUM_FRAMES - 1) // VAE_TEMPORAL_RATIO + 1 = (1-1)//4 + 1
LATENT_H = 60  # 480 // VAE_SPATIAL_RATIO (8)
LATENT_W = 90  # 720 // VAE_SPATIAL_RATIO (8)

NUM_CHANNELS_LATENTS = 16  # CogVideoX VAE latent channels (transformer in/out channels)

# Transformer (CogVideoX-5b)
NUM_ATTENTION_HEADS = 48
ATTENTION_HEAD_DIM = 64
TRANSFORMER_HIDDEN_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 3072
PATCH_SIZE = 2

# Text embedding dims
TEXT_EMBED_DIM = 4096  # T5 v1.1-XXL hidden size
TEXT_TOKEN_MAX_LEN = 226  # max_text_seq_length / tokenizer_max_length default
T5_VOCAB_SIZE = 32128

# Rotary positional embedding shape (CogVideoX-5b uses use_rotary_positional_embeddings=True)
ROTARY_NUM_PATCHES = (
    NUM_LATENT_FRAMES * (LATENT_H // PATCH_SIZE) * (LATENT_W // PATCH_SIZE)
)  # 1 * 30 * 45 = 1350

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load T5 v1.1-XXL text encoder from the text_encoder subfolder."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load CogVideoXTransformer3DModel from the transformer subfolder."""
    from diffusers import CogVideoXTransformer3DModel

    return CogVideoXTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLCogVideoX from the vae subfolder."""
    from diffusers import AutoencoderKLCogVideoX

    return AutoencoderKLCogVideoX.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Inputs for the T5 text encoder: [input_ids]."""
    input_ids = torch.randint(
        0, T5_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
    )
    return [input_ids]


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for the CogVideoX DiT transformer wrapper.

    Returns [hidden_states, encoder_hidden_states, timestep,
             image_rotary_emb_cos, image_rotary_emb_sin].
    """
    # CogVideoX hidden_states layout: (B, F, C, H, W); batch=2 for classifier-free guidance
    hidden_states = torch.randn(
        2,
        NUM_LATENT_FRAMES,
        NUM_CHANNELS_LATENTS,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        2, TEXT_TOKEN_MAX_LEN, TEXT_EMBED_DIM, dtype=dtype
    )
    timestep = torch.tensor([999, 999], dtype=torch.long)
    # 3D rotary positional embeddings: (num_patches, head_dim) per cos/sin
    image_rotary_emb_cos = torch.randn(
        ROTARY_NUM_PATCHES, ATTENTION_HEAD_DIM, dtype=dtype
    )
    image_rotary_emb_sin = torch.randn(
        ROTARY_NUM_PATCHES, ATTENTION_HEAD_DIM, dtype=dtype
    )
    return [
        hidden_states,
        encoder_hidden_states,
        timestep,
        image_rotary_emb_cos,
        image_rotary_emb_sin,
    ]


def load_vae_inputs(dtype: torch.dtype = DTYPE):
    """Inputs for the VAE decoder wrapper: [z (1,16,1,60,90)]."""
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
    """Expose only AutoencoderKLCogVideoX decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class CogVideoXTransformerWrapper(torch.nn.Module):
    """Simplify CogVideoXTransformer3DModel forward to tensor-only inputs/outputs.

    image_rotary_emb is reconstructed from (cos, sin) tensors so the wrapped forward
    has a flat tensor-only signature suitable for tracing/compilation.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        image_rotary_emb_cos,
        image_rotary_emb_sin,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=(image_rotary_emb_cos, image_rotary_emb_sin),
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Megatron column->row tensor-parallel shard specs for the T5 v1.1-XXL encoder.

    Weights are sharded only on the "model" (TP) axis; every other dim is
    replicated (None). The "batch" mesh axis is deliberately NOT used to slice
    weights -- it is 1/2/8 across QB/loudbox/galaxy and slicing a weight dim on
    it causes divisibility issues, so it only carries data-parallel replication.

      Column-parallel (q, k, v, wi_0, wi_1): ("model", None)
      Row-parallel    (o, wo):               (None, "model")
      Embeddings / norms / relative bias:    replicated
    T5 projections are bias-free, so only weights are sharded.
    """
    specs = {}

    if hasattr(encoder, "shared"):
        specs[encoder.shared.weight] = (None, None)

    stack = getattr(encoder, "encoder", encoder)

    if hasattr(stack, "embed_tokens"):
        specs[stack.embed_tokens.weight] = (None, None)

    blocks = getattr(stack, "block", None)
    if blocks is None:
        return specs

    for block in blocks:
        sa_layer = block.layer[0]
        sa = sa_layer.SelfAttention
        specs[sa.q.weight] = ("model", None)
        specs[sa.k.weight] = ("model", None)
        specs[sa.v.weight] = ("model", None)
        specs[sa.o.weight] = (None, "model")
        if getattr(sa, "relative_attention_bias", None) is not None:
            specs[sa.relative_attention_bias.weight] = (None, None)
        specs[sa_layer.layer_norm.weight] = (None,)

        ff_layer = block.layer[-1]
        dense = ff_layer.DenseReluDense
        if hasattr(dense, "wi_0"):
            specs[dense.wi_0.weight] = ("model", None)
            specs[dense.wi_1.weight] = ("model", None)
        else:
            specs[dense.wi.weight] = ("model", None)
        specs[dense.wo.weight] = (None, "model")
        specs[ff_layer.layer_norm.weight] = (None,)

    if hasattr(stack, "final_layer_norm"):
        specs[stack.final_layer_norm.weight] = (None,)

    return specs


def _causal_conv3d_weight(conv_module):
    """Resolve weight/bias tensors for a CogVideoXCausalConv3d (or plain Conv3d/Conv2d)."""
    inner = getattr(conv_module, "conv", conv_module)
    return inner.weight, getattr(inner, "bias", None)


def _shard_spatial_norm(norm, specs: dict) -> None:
    """Shard specs for a CogVideoXSpatialNorm3D module.

    Three principles that drive every choice below:

    (1) Replicated conv_shortcut-style residuals.
        The two 1x1 condition convs (`conv_y`, `conv_b`) feed the spatial
        norm's final combine: ``new_f = norm_layer(f) * conv_y + conv_b``.
        For that multiply/add to match without mid-op reshards, the
        condition convs are kept fully replicated — analogous to a
        residual conv_shortcut paired with a row-parallel `conv2` whose
        output is all-reduced to a replicated state.

    (2) Row-parallel biases are not sharded.
        Not directly applicable here (these are 1x1 convs we replicate
        outright), but the principle is mirrored: any tensor that lands
        on the "replicated" side of a Megatron pair has no axis sharded.

    (3) Norm statistics must be locally computable.
        The inner `norm_layer` is `nn.GroupNorm` with `num_groups=32`.
        CogVideoX VAE channel counts (128/256/512) are all multiples of
        32; with channel sharding along "model" (size 1/2/4 in our mesh
        set), each device holds an integer number of whole groups
        (group_size = channels/32, channels_per_device divisible by
        group_size). Per-group mean/var are therefore local and the
        replicated weight broadcasts correctly across the sharded
        channels — no cross-device statistic reduction is needed.
    """
    norm_layer = getattr(norm, "norm_layer", None)
    if norm_layer is not None:
        specs[norm_layer.weight] = (None,)
        if norm_layer.bias is not None:
            specs[norm_layer.bias] = (None,)

    for cond_conv_name in ("conv_y", "conv_b"):
        cond_conv = getattr(norm, cond_conv_name, None)
        if cond_conv is None:
            continue
        w, b = _causal_conv3d_weight(cond_conv)
        specs[w] = (None, None, None, None, None)
        if b is not None:
            specs[b] = (None,)


def _shard_resnet_block(block, specs: dict) -> None:
    """Shard specs for a CogVideoX VAE ResNet block.

    Megatron-style pair on `conv1`/`conv2`, sharding ONLY along the
    channel axis ("model") since tt-mlir Conv3D supports a single
    channel-axis sharding today. The three explicit rules baked in:

    (1) conv_shortcut is REPLICATED on every dim (weight and bias).
        `conv2` is row-parallel; its output is all-reduced back to a
        replicated state, so the residual add `conv2_out + shortcut_out`
        needs the shortcut already replicated to avoid an extra reshard.

    (2) Row-parallel biases are NOT sharded.
        `conv2.bias` is `(None,)`. A row-parallel layer all-reduces its
        partial output to a full result; if its bias were sharded along
        the same axis it would be summed N times by the all-reduce and
        produce wrong results.

    (3) Norm statistics must be locally computable.
        `norm1` / `norm2` can be either `nn.GroupNorm` (mid-block when
        `spatial_norm_dim=None`) or `CogVideoXSpatialNorm3D` (decoder,
        when `spatial_norm_dim` is set). In both cases the underlying
        normalization is GroupNorm with 32 groups, and channel counts
        (128/256/512) are multiples of 32, so any channel sharding into
        1/2/4 pieces keeps whole groups on each device — group
        mean/var stay local. We keep norm weights/biases replicated so
        the broadcast is correct regardless of activation shard state.

    Sharding summary:

      conv1 (col-parallel):
        weight ("model", None, None, None, None)
        bias   ("model",)                          # col-parallel bias IS sharded

      conv2 (row-parallel):
        weight (None, "model", None, None, None)
        bias   (None,)                             # (rule 2) never sharded

      conv_shortcut (replicated, rule 1):
        weight (None, None, None, None, None)
        bias   (None,)

      norm1 / norm2 (replicated, rule 3):
        weight / bias / spatial-norm sub-tensors: (None,) / (None,…)
    """
    norm1 = getattr(block, "norm1", None)
    if isinstance(norm1, torch.nn.GroupNorm):
        specs[norm1.weight] = (None,)
        if norm1.bias is not None:
            specs[norm1.bias] = (None,)
    elif norm1 is not None:
        _shard_spatial_norm(norm1, specs)

    # conv1: col-parallel — both weight and bias sharded on out-channel.
    w1, b1 = _causal_conv3d_weight(block.conv1)
    specs[w1] = ("model", None, None, None, None)
    if b1 is not None:
        specs[b1] = ("model",)

    norm2 = getattr(block, "norm2", None)
    if isinstance(norm2, torch.nn.GroupNorm):
        specs[norm2.weight] = (None,)
        if norm2.bias is not None:
            specs[norm2.bias] = (None,)
    elif norm2 is not None:
        _shard_spatial_norm(norm2, specs)

    # conv2: row-parallel — weight sharded on in-channel; bias NOT sharded
    # (rule 2: all-reduce would otherwise sum the bias N times).
    w2, b2 = _causal_conv3d_weight(block.conv2)
    specs[w2] = (None, "model", None, None, None)
    if b2 is not None:
        specs[b2] = (None,)

    # temb_proj is absent in CogVideoX decoder (temb_channels=0); if present
    # in another variant, output channel sharded along "model" only.
    if getattr(block, "temb_proj", None) is not None:
        specs[block.temb_proj.weight] = ("model", None)
        if block.temb_proj.bias is not None:
            specs[block.temb_proj.bias] = ("model",)

    # conv_shortcut: REPLICATED (rule 1) — must match conv2's all-reduced
    # output state so the residual add is a straight elementwise add.
    if getattr(block, "conv_shortcut", None) is not None:
        ws, bs = _causal_conv3d_weight(block.conv_shortcut)
        specs[ws] = (None, None, None, None, None)
        if bs is not None:
            specs[bs] = (None,)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKLCogVideoX (decoder-only path).

    Single-axis channel sharding on every Conv3D weight (axis "model").
    The three principles from `_shard_resnet_block` are applied
    consistently across the decoder:

    (1) Replicated conv_shortcut / spatial-norm condition convs on the
        residual / combine path so the merge with the row-parallel
        all-reduced output is a plain elementwise op.

    (2) Row-parallel biases (`conv2` in every resnet, `conv_out`) are
        kept replicated — bias must NOT be sharded along the same axis
        the all-reduce sums on.

    (3) Norm weights/biases (`GroupNorm` and the inner GroupNorm of
        `CogVideoXSpatialNorm3D`) are kept replicated. With
        `num_groups=32` and channel counts 128/256/512 the per-group
        statistics remain local across any "model" axis of size 1/2/4
        (whole groups per device).

    Per-tensor summary:

      Column-parallel (entry conv, first conv in resnet, upsampler conv):
        weight ("model", None, None, None, None), bias ("model",)
      Row-parallel   (second conv in resnet, exit conv):
        weight (None, "model", None, None, None), bias (None,)
      Replicated (conv_shortcut, spatial-norm condition convs,
                  all norm weights/biases):
        all dims None.
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

    for up_block in getattr(decoder, "up_blocks", []) or []:
        for resnet in getattr(up_block, "resnets", []) or []:
            _shard_resnet_block(resnet, specs)
        for upsampler in getattr(up_block, "upsamplers", []) or []:
            # CogVideoXUpsample3D.conv is nn.Conv2d → 4D weight; treat as
            # column-parallel on the single channel axis ("model"). For a
            # Conv3D fallback (5D weight), use the 5D col-parallel form.
            inner = getattr(upsampler, "conv", upsampler)
            w = inner.weight
            b = getattr(inner, "bias", None)
            if w.ndim == 5:
                specs[w] = ("model", None, None, None, None)
            else:
                specs[w] = ("model", None, None, None)
            if b is not None:
                specs[b] = ("model",)

    norm_out = getattr(decoder, "norm_out", None)
    if isinstance(norm_out, torch.nn.GroupNorm):
        specs[norm_out.weight] = (None,)
        if norm_out.bias is not None:
            specs[norm_out.bias] = (None,)
    elif norm_out is not None:
        _shard_spatial_norm(norm_out, specs)

    if hasattr(decoder, "conv_out"):
        w, b = _causal_conv3d_weight(decoder.conv_out)
        specs[w] = (None, "model", None, None, None)
        if b is not None:
            specs[b] = (None,)

    return specs


def shard_transformer_specs(transformer) -> dict:
    """Megatron column->row tensor-parallel shard specs for CogVideoXTransformer3DModel.

    Weights are sharded only on the "model" (TP) axis; every other dim is
    replicated (None). The "batch" mesh axis is deliberately NOT used to slice
    weights -- it is 1/2/8 across QB/loudbox/galaxy and slicing a weight dim on
    it causes divisibility issues, so it only carries data-parallel replication.

      Column-parallel (Q, K, V, FFN up, time_embedding up):  ("model", None)
      Row-parallel    (O, FFN down, time_embedding down):    (None, "model")

    Everything else is replicated:
      - patch_embed.proj / text_proj (block-stack entry boundaries),
      - the adaLN modulation linears in norm1/norm2 and norm_out (their chunked
        output is sliced into pieces -> replicate so the chunk-slices stay
        local), the proj_out exit projection,
      - the inner LayerNorm gamma/beta and norm_q/norm_k (act on the
        TP-replicated residual stream / shared across heads).
    """
    specs = {}

    # Patch / text embedding -- block-stack entry, replicated.
    if hasattr(transformer, "patch_embed"):
        pe = transformer.patch_embed
        if hasattr(pe, "proj"):
            proj = pe.proj
            specs[proj.weight] = (None,) * proj.weight.ndim
            if getattr(proj, "bias", None) is not None:
                specs[proj.bias] = (None,)
        if hasattr(pe, "text_proj"):
            specs[pe.text_proj.weight] = (None, None)
            if pe.text_proj.bias is not None:
                specs[pe.text_proj.bias] = (None,)

    # Time embedding -- Megatron column->row pair.
    if hasattr(transformer, "time_embedding"):
        te = transformer.time_embedding
        if hasattr(te, "linear_1"):
            specs[te.linear_1.weight] = ("model", None)
            if te.linear_1.bias is not None:
                specs[te.linear_1.bias] = ("model",)
        if hasattr(te, "linear_2"):
            specs[te.linear_2.weight] = (None, "model")
            if te.linear_2.bias is not None:
                specs[te.linear_2.bias] = (None,)

    for block in transformer.transformer_blocks:
        for norm_name in ("norm1", "norm2"):
            norm = getattr(block, norm_name, None)
            if norm is not None and hasattr(norm, "linear"):
                # Output (6 * embed_dim) is chunked into 6 modulation tensors;
                # replicate it so the chunk-slices stay local instead of
                # straddling shard boundaries.
                specs[norm.linear.weight] = (None, None)
                if norm.linear.bias is not None:
                    specs[norm.linear.bias] = (None,)
            if (
                norm is not None
                and hasattr(norm, "norm")
                and hasattr(norm.norm, "weight")
            ):
                # Inner LayerNorm acts on the TP-replicated residual stream.
                if norm.norm.weight is not None:
                    specs[norm.norm.weight] = (None,)
                if getattr(norm.norm, "bias", None) is not None:
                    specs[norm.norm.bias] = (None,)

        if hasattr(block, "attn1"):
            attn = block.attn1
            for proj_name in ("to_q", "to_k", "to_v"):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", None)
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            for norm_name in ("norm_q", "norm_k"):
                qk_norm = getattr(attn, norm_name, None)
                if (
                    qk_norm is not None
                    and hasattr(qk_norm, "weight")
                    and (qk_norm.weight is not None)
                ):
                    # LayerNorm(head_dim), shared across heads -> replicated.
                    specs[qk_norm.weight] = (None,)
                    if getattr(qk_norm, "bias", None) is not None:
                        specs[qk_norm.bias] = (None,)
            if hasattr(attn, "to_out") and attn.to_out is not None:
                out = attn.to_out
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = (None, "model")
                if target.bias is not None:
                    specs[target.bias] = (None,)

        if hasattr(block, "ff"):
            ff = block.ff
            if hasattr(ff.net[0], "proj"):
                specs[ff.net[0].proj.weight] = ("model", None)
                if ff.net[0].proj.bias is not None:
                    specs[ff.net[0].proj.bias] = ("model",)
            specs[ff.net[2].weight] = (None, "model")
            if ff.net[2].bias is not None:
                specs[ff.net[2].bias] = (None,)

    if hasattr(transformer, "norm_out") and hasattr(transformer.norm_out, "linear"):
        # Output (2 * embed_dim) is chunked into shift/scale; replicate it so
        # the chunk-slices stay local (same reasoning as the norm linears).
        lin = transformer.norm_out.linear
        specs[lin.weight] = (None, None)
        if lin.bias is not None:
            specs[lin.bias] = (None,)

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, None)
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
