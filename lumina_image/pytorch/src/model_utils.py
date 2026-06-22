# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Lumina-Image-2.0.

Model: Alpha-VLLM/Lumina-Image-2.0
Pipeline: diffusers.Lumina2Pipeline
Components:
  - text_encoder: Gemma2Model
  - transformer:  Lumina2Transformer2DModel
  - vae:          AutoencoderKL
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

LUMINA_REPO_ID = "Alpha-VLLM/Lumina-Image-2.0"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
# ---------------------------------------------------------------------------

HEIGHT = 1024
WIDTH = 1024

# AutoencoderKL spatial scaling factor (vae downsamples by 8)
VAE_SCALE_FACTOR = 8
LATENT_H = HEIGHT // VAE_SCALE_FACTOR  # 128
LATENT_W = WIDTH // VAE_SCALE_FACTOR  # 128

# Lumina2 transformer config: in_channels=16, patch_size=2, hidden_size=2304
NUM_CHANNELS_LATENTS = 16
PATCH_SIZE = 2
TEXT_EMBED_DIM = 2304

# Text encoder (Gemma-2): vocab=256000, hidden=2304
GEMMA_VOCAB_SIZE = 256000
MAX_SEQ_LEN = 256  # default Lumina2Pipeline max_sequence_length

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the Gemma2Model text encoder from the text_encoder subfolder."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load Lumina2Transformer2DModel from the transformer subfolder."""
    from diffusers import Lumina2Transformer2DModel

    return Lumina2Transformer2DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Synthetic input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype):
    """Inputs for Gemma2Model: [input_ids, attention_mask].

    Shapes match what Lumina2Pipeline.encode_prompt feeds into the text
    encoder when max_sequence_length=256: input_ids (1, 256) and
    attention_mask (1, 256), both int64.
    """
    input_ids = torch.randint(0, GEMMA_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_transformer_inputs(dtype: torch.dtype):
    """Inputs for Lumina2Transformer2DModel: [hidden_states, timestep,
    encoder_hidden_states, encoder_attention_mask].

    Shapes match a live forward through Lumina2Transformer2DModel:
      hidden_states          (1, 16, 128, 128)   — 1024x1024 image latent
      timestep               (1,)  == tensor([0.])
      encoder_hidden_states  (1, 256, 2304)      — Gemma-2 hidden_size
      encoder_attention_mask (1, 256)            — sourced from
                              load_text_encoder_inputs so it stays aligned
                              with the text encoder's attention_mask.
    """
    _, encoder_attention_mask = load_text_encoder_inputs(dtype)
    encoder_hidden_states = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=dtype)

    hidden_states = torch.randn(
        1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype
    )
    timestep = torch.zeros(1, dtype=torch.float32)
    return [hidden_states, timestep, encoder_hidden_states, encoder_attention_mask]


def load_vae_inputs(dtype: torch.dtype):
    """Inputs for VAEDecoderWrapper: [z]."""
    z = torch.randn(1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype)
    return [z]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class Gemma2TextEncoderWrapper(torch.nn.Module):
    """Run Gemma2Model as a stateless text encoder returning a plain tensor.

    Pins use_cache=False so no KV cache is built. With a cache, Gemma-2's
    sliding-window layer slices the value states with
    full_value_states[:, :, -sliding_window + 1 :, :] (sliding_window=4096),
    producing slice index -4095 which exceeds the tt-mlir slice bound of
    [-256, 255] (tenstorrent/tt-xla#4900). A single encode pass needs no
    cache, so disabling it removes the offending slice entirely. Also pins
    return_dict=False so graph capture sees a pure tensor (last_hidden_state).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


class Lumina2TransformerWrapper(torch.nn.Module):
    """Simplify Lumina2Transformer2DModel forward to return a plain tensor.

    The raw forward returns a Transformer2DModelOutput dataclass when
    return_dict=True. This wrapper pins return_dict=False and unwraps the
    single-element tuple so downstream graph capture sees a pure tensor.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL as (z) -> tensor.

    The default vae(z) runs encode+decode and returns a ModelOutput object.
    This wrapper calls decode directly and unwraps the output to a plain tensor.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for Gemma2Model.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, gate, up): ("model", "batch")
    Row-parallel   (o, down):            ("batch", "model")
    """
    specs = {encoder.embed_tokens.weight: (None, "batch")}

    for layer in encoder.layers:
        sa = layer.self_attn
        specs[sa.q_proj.weight] = ("model", "batch")
        specs[sa.k_proj.weight] = ("model", "batch")
        specs[sa.v_proj.weight] = ("model", "batch")
        specs[sa.o_proj.weight] = ("batch", "model")

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)
        specs[layer.pre_feedforward_layernorm.weight] = ("batch",)
        specs[layer.post_feedforward_layernorm.weight] = ("batch",)

    specs[encoder.norm.weight] = ("batch",)
    return specs


def _shard_lumina_block(block, specs: dict, has_adaln: bool) -> None:
    """Add shard specs for a single Lumina2TransformerBlock in-place.

    has_adaln distinguishes blocks that include the adaptive-norm path
    (noise_refiner / layers) from context_refiner blocks where norm1 is a
    plain RMSNorm without a learned modulation linear.
    """
    attn = block.attn
    specs[attn.to_q.weight] = ("model", "batch")
    specs[attn.to_k.weight] = ("model", "batch")
    specs[attn.to_v.weight] = ("model", "batch")
    specs[attn.to_out[0].weight] = ("batch", "model")

    ff = block.feed_forward
    specs[ff.linear_1.weight] = ("model", "batch")
    specs[ff.linear_3.weight] = ("model", "batch")
    specs[ff.linear_2.weight] = ("batch", "model")

    if has_adaln:
        specs[block.norm1.linear.weight] = ("model", "batch")
        specs[block.norm1.linear.bias] = ("model",)


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for Lumina2Transformer2DModel.

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, FFN up/gate): ("model", "batch")
    Row-parallel   (O, FFN down):           ("batch", "model")
    """
    specs = {
        transformer.x_embedder.weight: ("model", "batch"),
        transformer.x_embedder.bias: ("model",),
    }

    tce = transformer.time_caption_embed
    specs[tce.timestep_embedder.linear_1.weight] = ("model", "batch")
    specs[tce.timestep_embedder.linear_1.bias] = ("model",)
    specs[tce.timestep_embedder.linear_2.weight] = ("batch", "model")
    specs[tce.timestep_embedder.linear_2.bias] = ("batch",)
    specs[tce.caption_embedder[1].weight] = ("model", "batch")
    specs[tce.caption_embedder[1].bias] = ("model",)

    for block in transformer.noise_refiner:
        _shard_lumina_block(block, specs, has_adaln=True)
    for block in transformer.context_refiner:
        _shard_lumina_block(block, specs, has_adaln=False)
    for block in transformer.layers:
        _shard_lumina_block(block, specs, has_adaln=True)

    specs[transformer.norm_out.linear_1.weight] = ("model", "batch")
    specs[transformer.norm_out.linear_1.bias] = ("model",)
    specs[transformer.norm_out.linear_2.weight] = ("batch", "model")
    specs[transformer.norm_out.linear_2.bias] = ("batch",)

    return specs


def _shard_resnet_block_megatron(block, specs: dict) -> None:
    """Megatron-style channel sharding within a diffusers ResnetBlock2D.

    Pairs the block's two convs so that only one all_reduce is needed per block:
      conv1 → column-parallel: weight (out, in, kH, kW) sharded on dim 0
              → ("model", None, None, None); bias sharded → ("model",)
      conv2 → row-parallel:    weight (out, out, kH, kW) sharded on dim 1
              → (None, "model", None, None); bias replicated(added once
              after all_reduce).
    conv_shortcut consumes the original replicated input and stays replicated
    so the residual add x + h matches the full output of conv2's all_reduce.
    """
    specs[block.conv1.weight] = ("model", None, None, None)
    if block.conv1.bias is not None:
        specs[block.conv1.bias] = ("model",)

    specs[block.conv2.weight] = (None, "model", None, None)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKL (decoder path used by VAEDecoderWrapper).

    Strategy: Megatron-style channel sharding along a single mesh axis —
    "model" — only. Sharding along both mesh axes would break on multi-axis
    meshes (e.g. Galaxy (8, 4)) because Conv2D/Conv3D in tt-mlir can currently
    only be partitioned on the channel dim. Within each ResnetBlock2D the
    two convs alternate column-parallel → row-parallel:

      ResnetBlock2D.conv1   → column-parallel  ("model", None, None, None)
                              bias              → ("model",)
      ResnetBlock2D.conv2   → row-parallel     (None, "model", None, None)
                              bias (b2)         → replicated (omitted)
      conv_shortcut, Up/Downsample.conv, conv_in, conv_out  → replicated
      GroupNorm, conv_norm_out, attention.group_norm        → replicated

    The mid-block Attention follows the standard Megatron MLP pattern, also
    single-axis on "model":
      to_q / to_k / to_v   → column-parallel  ("model", None), bias ("model",)
      to_out[0]            → row-parallel     (None, "model"), bias replicated

    conv_in (16 → 512) and conv_out (128 → 3) are replicated so each resnet
    pair starts from a full input; conv_out's 3 output channels also do not
    divide cleanly across the model axis.

    GroupNorm affine weights are not sharded: group statistics need the full
    channel dim, so sharding the norm's channel-aligned weight would
    desynchronize the local-stat math. The runtime all_gathers the channel-
    sharded activation before each GroupNorm, runs the norm on a replicated
    tensor, then the next row-parallel conv computes partial sums + all_reduce
    to bring the residual output back to replicated for the shortcut add.
    """
    specs: dict = {}
    decoder = vae.decoder

    # conv_in: replicated (feeds full input into first column-parallel conv)
    # → omitted from specs

    for up_block in decoder.up_blocks:
        for resnet in up_block.resnets:
            _shard_resnet_block_megatron(resnet, specs)
        # upsampler convs: standalone, replicated → omitted

    mid = decoder.mid_block
    for resnet in mid.resnets:
        _shard_resnet_block_megatron(resnet, specs)
    for attn in mid.attentions:
        specs[attn.to_q.weight] = ("model", None)
        specs[attn.to_q.bias] = ("model",)
        specs[attn.to_k.weight] = ("model", None)
        specs[attn.to_k.bias] = ("model",)
        specs[attn.to_v.weight] = ("model", None)
        specs[attn.to_v.bias] = ("model",)
        specs[attn.to_out[0].weight] = (None, "model")
        # to_out[0].bias: replicated (added once after all_reduce)

    # conv_out: replicated → omitted from specs

    return specs
