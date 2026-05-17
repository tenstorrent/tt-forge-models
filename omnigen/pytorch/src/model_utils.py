# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for OmniGen.

Model: Shitao/OmniGen-v1-diffusers
Components:
  - transformer: OmniGenTransformer2DModel (embeds text tokens internally
                 via a LLaMA-style `embed_tokens`; no separate text encoder)
  - vae:         AutoencoderKL (4-channel SD-style latents)

The pipeline also carries a FlowMatchEulerDiscreteScheduler and a
LlamaTokenizer, but those have no learnable weights and are not exposed
as variants here.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "Shitao/OmniGen-v1-diffusers"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
#
# Matches the live pipeline shapes observed at runtime for the default
# 1024x1024 generation (height=1024, width=1024):
#
#   hidden_states  : [2, 4, 128, 128]    # CFG_BATCH x C_lat x H_lat x W_lat
#   timestep       : [2]                 # value tensor([0., 0.])
#   input_ids      : [2, 186]            # tokenized prompt length
#   attention_mask : [2, 4283, 4283]     # TOTAL_SEQ_LEN x TOTAL_SEQ_LEN
#   position_ids   : [2, 4283]
#
# Derivation:
#   VAE spatial_compression = 8         → latent_h/w = 1024 // 8 = 128
#   patch_size = 2                      → post-patch grid = 64 x 64
#   num_image_tokens = 64 * 64 = 4096
#   total_seq = TEXT_SEQ_LEN + 1 (time_token) + num_image_tokens
#             = 186 + 1 + 4096          = 4283
# ---------------------------------------------------------------------------

IMAGE_H = 1024
IMAGE_W = 1024
VAE_SCALE_FACTOR = 8
LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 128
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 128

LATENT_CHANNELS = 4  # AutoencoderKL latent channels
TRANSFORMER_IN_CHANNELS = 4  # OmniGenTransformer2DModel.in_channels
PATCH_SIZE = 2

POST_PATCH_H = LATENT_H // PATCH_SIZE  # 64
POST_PATCH_W = LATENT_W // PATCH_SIZE  # 64
NUM_IMAGE_TOKENS = POST_PATCH_H * POST_PATCH_W  # 4096

TEXT_SEQ_LEN = 186  # observed from OmniGenMultiModalProcessor on default prompt
TIME_TOKEN_LEN = 1
TOTAL_SEQ_LEN = TEXT_SEQ_LEN + TIME_TOKEN_LEN + NUM_IMAGE_TOKENS  # 4283

# OmniGen transformer config defaults
VOCAB_SIZE = 32064
PAD_TOKEN_ID = 32000

# Classifier-free guidance doubles the batch (cond + uncond).
CFG_BATCH = 2

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load OmniGenTransformer2DModel from the transformer subfolder."""
    from diffusers import OmniGenTransformer2DModel

    return OmniGenTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Component input builders
# ---------------------------------------------------------------------------


def load_transformer_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic inputs for OmniGenTransformer2DModel.

    Returns [hidden_states, timestep, input_ids, attention_mask, position_ids].
    `input_img_latents` and `input_image_sizes` are passed through the
    wrapper as empty/static structures (text-only generation path).
    """
    hidden_states = torch.randn(
        CFG_BATCH,
        TRANSFORMER_IN_CHANNELS,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    # Hardcoded to match the live pipeline trace at num_inference_steps=1,
    # where the single scheduler timestep evaluates to 0 for both CFG rows.
    timestep = torch.zeros(CFG_BATCH, dtype=dtype)
    input_ids = torch.randint(
        0, VOCAB_SIZE, (CFG_BATCH, TEXT_SEQ_LEN), dtype=torch.long
    )
    attention_mask = torch.ones(CFG_BATCH, TOTAL_SEQ_LEN, TOTAL_SEQ_LEN, dtype=dtype)
    position_ids = (
        torch.arange(TOTAL_SEQ_LEN, dtype=torch.long)
        .unsqueeze(0)
        .expand(CFG_BATCH, -1)
        .contiguous()
    )
    return [hidden_states, timestep, input_ids, attention_mask, position_ids]


def load_vae_decoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic latent input for VAEDecoderWrapper: [z (1, 4, 16, 16)]."""
    z = torch.randn(
        1,
        LATENT_CHANNELS,
        LATENT_H,
        LATENT_W,
        dtype=dtype,
    )
    return [z]


def load_vae_encoder_inputs(dtype: torch.dtype = DTYPE):
    """Synthetic pixel-space input for VAEEncoderWrapper: [x (1, 3, 128, 128)].

    Matches the conditioning-image path in `pipeline_omnigen.encode_input_images`,
    which feeds RGB tensors in [-1, 1] through `vae.encode(...)`.
    """
    x = torch.randn(1, 3, IMAGE_H, IMAGE_W, dtype=dtype)
    return [x]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKL decoder as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class VAEEncoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKL encoder as (x) -> tensor.

    Returns `mode()` (the posterior mean) instead of `sample()` so the
    forward is deterministic and traceable — `sample()` would introduce
    RNG and a non-tensor `generator` argument.

    Applies `vae.config.scaling_factor` to match what the OmniGen pipeline
    does at pipeline_omnigen.py:190 (`...sample().mul_(scaling_factor)`).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.scaling_factor = vae.config.scaling_factor

    def forward(self, x):
        posterior = self.vae.encode(x, return_dict=False)[0]
        return posterior.mode() * self.scaling_factor


class OmniGenTransformerWrapper(torch.nn.Module):
    """Simplify OmniGenTransformer2DModel forward to tensor-only inputs/outputs.

    `input_img_latents` and `input_image_sizes` are non-tensor structures and
    are fixed to the text-only generation case (empty list / empty dict) so
    the wrapper can be traced/exported with positional tensor arguments.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        input_ids,
        attention_mask,
        position_ids,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            input_ids=input_ids,
            input_img_latents=[],
            input_image_sizes={},
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def _shard_resnet_block_2d(block, specs: dict) -> None:
    """Shard specs for an AutoencoderKL ResNet block (Conv2d, 4D weights).

    Megatron-style pair (conv1 col-parallel, conv2 row-parallel) with a
    replicated shortcut so the residual add works without re-sharding:

      conv1 (col-parallel):
        weight ("model", None, None, None)   # shard Cout
        bias   ("model",)

      conv2 (row-parallel):
        weight (None, "model", None, None)   # shard Cin
        bias   (None,)                       # replicated after all_reduce

      conv_shortcut (replicated): all dims None.
    """
    if hasattr(block, "norm1"):
        specs[block.norm1.weight] = (None,)
        if block.norm1.bias is not None:
            specs[block.norm1.bias] = (None,)

    specs[block.conv1.weight] = ("model", None, None, None)
    if block.conv1.bias is not None:
        specs[block.conv1.bias] = ("model",)

    if hasattr(block, "norm2"):
        specs[block.norm2.weight] = (None,)
        if block.norm2.bias is not None:
            specs[block.norm2.bias] = (None,)

    specs[block.conv2.weight] = (None, "model", None, None)
    if block.conv2.bias is not None:
        specs[block.conv2.bias] = (None,)

    if getattr(block, "conv_shortcut", None) is not None:
        specs[block.conv_shortcut.weight] = (None, None, None, None)
        if block.conv_shortcut.bias is not None:
            specs[block.conv_shortcut.bias] = (None,)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKL (decoder-only path).

    For Conv2d weights (out_ch, in_ch, H, W) only one channel dim is sharded
    (along "model"):
      Column-parallel (entry conv, first conv in resnet, upsampler conv):
        weight ("model", None, None, None), bias ("model",)
      Row-parallel   (second conv in resnet, exit conv):
        weight (None, "model", None, None), bias (None,)
      Replicated (conv_shortcut): all dims None for both weight and bias.
    Norm / attention specs mirror the transformer conventions.
    """
    specs = {}

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        return specs

    if hasattr(decoder, "conv_in"):
        specs[decoder.conv_in.weight] = ("model", None, None, None)
        if decoder.conv_in.bias is not None:
            specs[decoder.conv_in.bias] = ("model",)

    mid_block = getattr(decoder, "mid_block", None)
    if mid_block is not None:
        for resnet in getattr(mid_block, "resnets", []) or []:
            _shard_resnet_block_2d(resnet, specs)
        for attn in getattr(mid_block, "attentions", []) or []:
            if attn is None:
                continue
            if hasattr(attn, "group_norm") and attn.group_norm is not None:
                specs[attn.group_norm.weight] = (None,)
                if attn.group_norm.bias is not None:
                    specs[attn.group_norm.bias] = (None,)
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
            _shard_resnet_block_2d(resnet, specs)
        for upsampler in getattr(up_block, "upsamplers", []) or []:
            specs[upsampler.conv.weight] = ("model", None, None, None)
            if upsampler.conv.bias is not None:
                specs[upsampler.conv.bias] = ("model",)

    if hasattr(decoder, "conv_norm_out"):
        specs[decoder.conv_norm_out.weight] = (None,)
        if decoder.conv_norm_out.bias is not None:
            specs[decoder.conv_norm_out.bias] = (None,)

    if hasattr(decoder, "conv_out"):
        specs[decoder.conv_out.weight] = (None, "model", None, None)
        if decoder.conv_out.bias is not None:
            specs[decoder.conv_out.bias] = (None,)

    return specs


def shard_vae_encoder_specs(vae) -> dict:
    """Shard specs for AutoencoderKL (encoder-only path, plus `quant_conv`).

    Same Megatron pair pattern as the decoder:
      Column-parallel (entry conv, first conv in resnet, downsampler conv):
        weight ("model", None, None, None), bias ("model",)
      Row-parallel   (second conv in resnet, exit conv):
        weight (None, "model", None, None), bias (None,)
      Replicated (conv_shortcut, norms, post-encoder quant_conv): all None.

    `quant_conv` is a small 1x1 Conv2d (`2*latent_channels -> 2*latent_channels`)
    that re-projects the encoder's (mean, logvar) parameters. Sharding it
    would force resharding right before the distribution is constructed,
    so it stays replicated.
    """
    specs = {}

    encoder = getattr(vae, "encoder", None)
    if encoder is None:
        return specs

    if hasattr(encoder, "conv_in"):
        specs[encoder.conv_in.weight] = ("model", None, None, None)
        if encoder.conv_in.bias is not None:
            specs[encoder.conv_in.bias] = ("model",)

    for down_block in getattr(encoder, "down_blocks", []) or []:
        for resnet in getattr(down_block, "resnets", []) or []:
            _shard_resnet_block_2d(resnet, specs)
        for downsampler in getattr(down_block, "downsamplers", []) or []:
            specs[downsampler.conv.weight] = ("model", None, None, None)
            if downsampler.conv.bias is not None:
                specs[downsampler.conv.bias] = ("model",)

    mid_block = getattr(encoder, "mid_block", None)
    if mid_block is not None:
        for resnet in getattr(mid_block, "resnets", []) or []:
            _shard_resnet_block_2d(resnet, specs)
        for attn in getattr(mid_block, "attentions", []) or []:
            if attn is None:
                continue
            if hasattr(attn, "group_norm") and attn.group_norm is not None:
                specs[attn.group_norm.weight] = (None,)
                if attn.group_norm.bias is not None:
                    specs[attn.group_norm.bias] = (None,)
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

    if hasattr(encoder, "conv_norm_out"):
        specs[encoder.conv_norm_out.weight] = (None,)
        if encoder.conv_norm_out.bias is not None:
            specs[encoder.conv_norm_out.bias] = (None,)

    if hasattr(encoder, "conv_out"):
        specs[encoder.conv_out.weight] = (None, "model", None, None)
        if encoder.conv_out.bias is not None:
            specs[encoder.conv_out.bias] = (None,)

    quant_conv = getattr(vae, "quant_conv", None)
    if quant_conv is not None:
        specs[quant_conv.weight] = (None, None, None, None)
        if quant_conv.bias is not None:
            specs[quant_conv.bias] = (None,)

    return specs


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for OmniGenTransformer2DModel.

    Column-parallel (Q, K, V, gate_up_proj): ("model", "batch")
    Row-parallel   (attn out, down_proj):    ("batch", "model")
    """
    specs = {}

    patch = getattr(transformer, "patch_embedding", None)
    if patch is not None:
        for proj_name in ("output_image_proj", "input_image_proj"):
            proj = getattr(patch, proj_name, None)
            if proj is None:
                continue
            specs[proj.weight] = ("batch", None, None, None)
            if proj.bias is not None:
                specs[proj.bias] = ("batch",)

    if hasattr(transformer, "embed_tokens"):
        specs[transformer.embed_tokens.weight] = (None, "batch")

    for block in transformer.layers:
        attn = block.self_attn
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

        mlp = block.mlp
        specs[mlp.gate_up_proj.weight] = ("model", "batch")
        if mlp.gate_up_proj.bias is not None:
            specs[mlp.gate_up_proj.bias] = ("model",)
        specs[mlp.down_proj.weight] = ("batch", "model")
        if mlp.down_proj.bias is not None:
            specs[mlp.down_proj.bias] = ("batch",)

        specs[block.input_layernorm.weight] = (None,)
        specs[block.post_attention_layernorm.weight] = (None,)

    if hasattr(transformer, "norm"):
        specs[transformer.norm.weight] = (None,)

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
