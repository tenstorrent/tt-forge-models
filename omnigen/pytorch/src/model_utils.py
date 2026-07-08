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

import os

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "Shitao/OmniGen-v1-diffusers"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Diagnostic toggles (env-gated; unset => no effect on normal runs).
#
#   OMNIGEN_NUM_LAYERS=<N>   truncate transformer to first N blocks (isolate
#                            per-op vs cumulative-depth precision loss).
#   OMNIGEN_ATTN_FP32=1      run each block's self-attention in fp32.
#   OMNIGEN_MLP_FP32=1       run each block's MLP in fp32.
#   OMNIGEN_NORM_FP32=1      run RMSNorms in fp32.
#   OMNIGEN_RESIDUAL_FP32=1  keep the whole block stack + residual in fp32.
#   OMNIGEN_EAGER_ATTN=1     replace fused SDPA with manual softmax attention.
#   OMNIGEN_FP16_MODEL=1     run the whole model (weights+inputs) in fp16.
#   OMNIGEN_RELU_MLP=1       replace SiLU with ReLU (exact) in the MLP.
#   OMNIGEN_SILU_FP32=1      run SiLU in fp32 (typecast-wrapped).
#   OMNIGEN_SILU_EXP=1       decompose SiLU as x/(1+exp(-x)) (no sigmoid op).
# ---------------------------------------------------------------------------

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
    """Load OmniGenTransformer2DModel from the transformer subfolder.

    The fused ``gate_up_proj`` of every block is split into separate
    ``gate_proj`` / ``up_proj`` linears (see ``_SplitGateUpFeedForward``) so the
    MLP can use the standard column->row Megatron pattern under tensor
    parallelism. This is a numerically-identical rewrite on CPU, so the golden
    (CPU) result is unchanged.
    """
    from diffusers import OmniGenTransformer2DModel

    # Diagnostic: run the entire model in fp16 (weights + activations). fp16 has
    # 10 mantissa bits vs bf16's 7, so if the PCC drop is mantissa precision
    # accumulating over depth, fp16 recovers most of it at the same 16-bit width.
    if os.environ.get("OMNIGEN_FP16_MODEL") == "1":
        dtype = torch.float16

    transformer = OmniGenTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()

    for block in transformer.layers:
        block.mlp = _SplitGateUpFeedForward(block.mlp)

    _apply_diagnostic_toggles(transformer, dtype)

    return transformer


def _apply_diagnostic_toggles(transformer, dtype):
    """Apply env-gated precision/isolation toggles (see module header)."""
    num_layers = os.environ.get("OMNIGEN_NUM_LAYERS")
    if num_layers:
        n = int(num_layers)
        transformer.layers = transformer.layers[:n]

    attn_fp32 = os.environ.get("OMNIGEN_ATTN_FP32") == "1"
    mlp_fp32 = os.environ.get("OMNIGEN_MLP_FP32") == "1"
    norm_fp32 = os.environ.get("OMNIGEN_NORM_FP32") == "1"
    residual_fp32 = os.environ.get("OMNIGEN_RESIDUAL_FP32") == "1"
    eager_attn = os.environ.get("OMNIGEN_EAGER_ATTN") == "1"
    relu_mlp = os.environ.get("OMNIGEN_RELU_MLP") == "1"
    silu_fp32 = os.environ.get("OMNIGEN_SILU_FP32") == "1"

    if relu_mlp:
        # Replace SiLU (x*sigmoid(x), SFPU transcendental) with ReLU (exact,
        # max(x,0)). If PCC jumps, the sigmoid SFPU approximation is the culprit.
        for block in transformer.layers:
            block.mlp.activation_fn = torch.nn.ReLU()

    if silu_fp32:
        # Run only the SiLU in fp32 (typecast bf16->f32 -> silu -> f32->bf16).
        # This mirrors the proposed tt-mlir fix: forcing fp32 makes tt-metal's
        # unary_impl set fp32_dest_acc_en=true, selecting the accurate sigmoid
        # (accurate exp + 2-iter reciprocal) instead of the coarse bf16 path.
        for block in transformer.layers:
            block.mlp.activation_fn = _Fp32Activation(block.mlp.activation_fn, dtype)

    if os.environ.get("OMNIGEN_SILU_EXP") == "1":
        # Decompose SiLU as x / (1 + exp(-x)) using explicit exp + reciprocal
        # instead of the fused sigmoid SFPU op. If PCC jumps, the sigmoid op
        # itself is inaccurate and decomposing it in tt-mlir is the fix.
        for block in transformer.layers:
            block.mlp.activation_fn = _ExpSiLU()

    if eager_attn:
        # Replace fused F.scaled_dot_product_attention with manual
        # matmul -> softmax -> matmul, which lowers to ttnn.softmax (no
        # exp-approx) instead of the fused SDPA kernel's approximate exp.
        for block in transformer.layers:
            block.self_attn.set_processor(_EagerOmniGenAttnProcessor())

    for block in transformer.layers:
        if attn_fp32:
            block.self_attn = _Fp32Module(block.self_attn, dtype)
        if mlp_fp32:
            block.mlp = _Fp32Module(block.mlp, dtype)
        if norm_fp32:
            block.input_layernorm = _Fp32Module(block.input_layernorm, dtype)
            block.post_attention_layernorm = _Fp32Module(
                block.post_attention_layernorm, dtype
            )

    if residual_fp32:
        # Keep the whole block stack (incl. the residual adds) in fp32 by
        # chaining fp32 blocks without downcasting between them. Isolates the
        # bf16 residual-accumulation error from per-op compute precision.
        transformer.layers = torch.nn.ModuleList(
            [_Fp32ChainBlock(b) for b in transformer.layers]
        )


class _Fp32Module(torch.nn.Module):
    """Diagnostic wrapper: run `mod` in fp32, cast inputs up / outputs back.

    Upcasts floating-point tensor args/kwargs to fp32, runs the fp32-weighted
    module, then downcasts floating-point outputs to `out_dtype`. Used to test
    whether a given component's bf16 precision is what caps PCC.
    """

    def __init__(self, mod, out_dtype):
        super().__init__()
        self.mod = mod.float()
        self._out_dtype = out_dtype

    @staticmethod
    def _map(x, cast):
        if torch.is_tensor(x):
            return cast(x) if x.is_floating_point() else x
        if isinstance(x, (list, tuple)):
            return type(x)(_Fp32Module._map(i, cast) for i in x)
        if isinstance(x, dict):
            return {k: _Fp32Module._map(v, cast) for k, v in x.items()}
        return x

    def forward(self, *args, **kwargs):
        up = lambda x: x.float()
        args = tuple(self._map(a, up) for a in args)
        kwargs = {k: self._map(v, up) for k, v in kwargs.items()}
        out = self.mod(*args, **kwargs)
        return self._map(out, lambda x: x.to(self._out_dtype))


class _Fp32ChainBlock(torch.nn.Module):
    """Diagnostic: run a transformer block in fp32 WITHOUT downcasting output.

    Upcasts all floating inputs to fp32 and returns fp32, so chaining these
    across every block keeps the residual stream (and its adds) in fp32 the
    whole way through the stack — unlike `_Fp32Module`, which downcasts at each
    boundary and leaves the residual in bf16.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block.float()

    def forward(self, *args, **kwargs):
        up = lambda x: x.float()
        args = tuple(_Fp32Module._map(a, up) for a in args)
        kwargs = {k: _Fp32Module._map(v, up) for k, v in kwargs.items()}
        return self.block(*args, **kwargs)


class _ExpSiLU(torch.nn.Module):
    """SiLU decomposed as x / (1 + exp(-x)) — avoids the fused sigmoid op."""

    def forward(self, x):
        return x * torch.reciprocal(1.0 + torch.exp(-x))


class _Fp32Activation(torch.nn.Module):
    """Run an elementwise activation in fp32: cast input up, act, cast back.

    Mirrors a tt-mlir typecast-wrapped activation so the op runs with
    fp32_dest_acc_en=true (accurate SFPU path).
    """

    def __init__(self, act, out_dtype):
        super().__init__()
        self.act = act
        self._out_dtype = out_dtype

    def forward(self, x):
        return self.act(x.float()).to(self._out_dtype)


class _EagerOmniGenAttnProcessor:
    """Diagnostic attention processor: manual matmul -> softmax -> matmul.

    Mirrors diffusers' OmniGenAttnProcessor2_0 exactly, except it replaces the
    fused ``F.scaled_dot_product_attention`` with an explicit implementation so
    the softmax lowers to ``ttnn.softmax`` (numerically stable, no exp-approx)
    rather than the fused SDPA kernel's approximate exp.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        import math

        from diffusers.models.embeddings import apply_rotary_emb

        bsz, q_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        head_dim = query_dim // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, kv_heads, head_dim).transpose(1, 2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real_unbind_dim=-2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        if attention_mask is not None:
            # F.sdpa adds a floating attn_mask to the scores; match that.
            scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attn_weights, value)

        hidden_states = hidden_states.transpose(1, 2).type_as(query)
        hidden_states = hidden_states.reshape(bsz, q_len, attn.out_dim)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


class _SplitGateUpFeedForward(torch.nn.Module):
    """OmniGenFeedForward with the fused ``gate_up_proj`` split in two.

    diffusers' ``OmniGenFeedForward`` runs a single fused projection and then
    ``gate, up = up_states.chunk(2, dim=-1)``. Column-sharding that fused output
    scrambles the chunk (the partitioner does not all-gather before the chunk),
    which forces the projection to be row-parallel and adds an extra all-reduce
    per block. Splitting into two independent linears lets each be
    column-parallel, so the MLP follows the standard column->row pattern with a
    single all-reduce (in ``down_proj``) and each up-projection's contraction
    stays fully in fp32 — matching single-chip precision.

    Weight split mirrors ``chunk(2, dim=-1)``: rows ``[:intermediate]`` are the
    gate, rows ``[intermediate:]`` are the up projection.
    """

    def __init__(self, fused_mlp):
        super().__init__()
        gate_up = fused_mlp.gate_up_proj
        hidden_size = gate_up.in_features
        intermediate_size = gate_up.out_features // 2
        has_bias = gate_up.bias is not None

        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=has_bias)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=has_bias)
        self.down_proj = fused_mlp.down_proj
        self.activation_fn = fused_mlp.activation_fn

        self.gate_proj.weight = torch.nn.Parameter(
            gate_up.weight[:intermediate_size].detach().clone()
        )
        self.up_proj.weight = torch.nn.Parameter(
            gate_up.weight[intermediate_size:].detach().clone()
        )
        if has_bias:
            self.gate_proj.bias = torch.nn.Parameter(
                gate_up.bias[:intermediate_size].detach().clone()
            )
            self.up_proj.bias = torch.nn.Parameter(
                gate_up.bias[intermediate_size:].detach().clone()
            )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


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
    # Keep input dtype in sync with the OMNIGEN_FP16_MODEL diagnostic toggle.
    if os.environ.get("OMNIGEN_FP16_MODEL") == "1":
        dtype = torch.float16

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
    """Tensor-parallel shard specs for OmniGenTransformer2DModel.

    Megatron tensor parallelism on the "model" mesh axis only; the "batch"
    axis is reserved for data parallelism and never appears in a weight spec.

    Per block (LLaMA/Phi-3-style: RMSNorm -> attention -> RMSNorm -> MLP):
      - attention to_q/to_k/to_v: column-parallel ("model", None). This shards
        the 32 heads (head_dim stays whole), which is safe under the
        view(..., heads, head_dim) reshape since the shard boundary lands on a
        head boundary (768 = 8 heads * 96).
      - attention to_out:         row-parallel   (None, "model").
      - mlp.gate_proj / mlp.up_proj: column-parallel ("model", None). The fused
        gate_up_proj is split at load time (see _SplitGateUpFeedForward) so each
        projection can be column-sharded without scrambling the chunk; their
        contraction over hidden stays fully in fp32 (no reduction), matching
        single-chip precision.
      - mlp.down_proj:            row-parallel   (None, "model").

    The residual stream, RMSNorm weights, embeddings, patch embedding and
    proj_out stay replicated so every RMSNorm reduces over the full (un-sharded)
    hidden dim and the row-parallel all-reduces return the residual replicated.
    """
    specs = {}

    # Diagnostic fp32 wrappers keep the real submodule under `.mod`; unwrap so
    # the projections still get sharded (heads split), avoiding the giant
    # unsharded attention tensor that OOMs.
    def _unwrap(m):
        return m.mod if isinstance(m, _Fp32Module) else m

    for block in transformer.layers:
        if isinstance(block, _Fp32ChainBlock):
            block = block.block
        attn = _unwrap(block.self_attn)
        for proj_name in ("to_q", "to_k", "to_v"):
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                specs[proj.weight] = ("model", None)
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)
        if hasattr(attn, "to_out"):
            out = attn.to_out
            target = (
                out[0]
                if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                else out
            )
            specs[target.weight] = (None, "model")
            if target.bias is not None:
                specs[target.bias] = (None,)

        mlp = _unwrap(block.mlp)
        if hasattr(mlp, "gate_proj"):
            for proj_name in ("gate_proj", "up_proj"):
                proj = getattr(mlp, proj_name)
                specs[proj.weight] = ("model", None)
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)
            specs[mlp.down_proj.weight] = (None, "model")
            if mlp.down_proj.bias is not None:
                specs[mlp.down_proj.bias] = (None,)

    return specs
