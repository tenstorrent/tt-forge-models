# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, forward-signature wrappers, and SPMD shard specifications for
FLUX.2-dev (black-forest-labs/FLUX.2-dev).

FLUX.2-dev is a rectified-flow text-to-image model composed of three independently
loadable components:
  - text_encoder : Mistral3ForConditionalGeneration (Mistral-Small-3.x, ~24 B)
                   The pipeline stacks intermediate hidden states (layers 10/20/30)
                   into a 3*5120 = 15360-dim conditioning sequence.
  - transformer  : Flux2Transformer2DModel (MM-DiT, 8 dual + 48 single blocks,
                   inner_dim=6144, ~32 B). Guidance-distilled.
  - vae          : AutoencoderKLFlux2 (32-channel latent KL autoencoder, ~0.1 B)

Each is large enough (transformer, text_encoder) to require tensor parallelism on
hardware; the VAE fits on a single chip.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

FLUX2_REPO_ID = "black-forest-labs/FLUX.2-dev"
DTYPE = torch.bfloat16

# Pipeline stacks these text-encoder hidden-state layers into the 15360-dim
# conditioning sequence (see Flux2Pipeline._get_mistral_3_small_prompt_embeds).
TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)

# ---------------------------------------------------------------------------
# Inference shape constants
#
# Kept deliberately small for the hardware bringup tests: a 256x256 image gives
# 256 packed image tokens, paired with a 128-token text sequence. These can be
# enlarged once the model compiles. Latent math mirrors Flux2Pipeline:
#   latent_hw = 2 * (image_hw // (vae_scale_factor * 2))
#   packed image tokens = (latent_h // 2) * (latent_w // 2)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256
VAE_SCALE_FACTOR = 8

# Latent grid after VAE compression (before transformer 2x2 packing).
LATENT_HW = 2 * (IMAGE_SIZE // (VAE_SCALE_FACTOR * 2))  # 32 for 256px
VAE_LATENT_CHANNELS = 32

# Transformer-side packed shapes.
DIT_IN_CHANNELS = 128  # vae latent_channels(32) * 2*2 packing
DIT_JOINT_ATTENTION_DIM = 15360  # 3 * 5120 stacked Mistral hidden states
DIT_INNER_DIM = 6144
DIT_NUM_ROPE_AXES = 4  # axes_dims_rope = (32, 32, 32, 32)
IMAGE_SEQ_LEN = (LATENT_HW // 2) * (LATENT_HW // 2)  # 256 packed image tokens
TEXT_SEQ_LEN = 128  # text tokens fed to the DiT (<= max_sequence_length)

# Mistral-3 text encoder dims.
TEXT_HIDDEN = 5120
TEXT_VOCAB_SIZE = 131072


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load Flux2Transformer2DModel (the MM-DiT) from the transformer subfolder."""
    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the Mistral3 text encoder from the text_encoder subfolder."""
    from transformers import Mistral3ForConditionalGeneration

    return Mistral3ForConditionalGeneration.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKLFlux2 from the vae subfolder."""
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — flatten component forward signatures to positional tensors,
# mirroring exactly what Flux2Pipeline does at each call site.
# ---------------------------------------------------------------------------


class Flux2TransformerWrapper(torch.nn.Module):
    """Expose the DiT forward as pure positional tensors, returning the sample."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]


class Mistral3PromptEmbedWrapper(torch.nn.Module):
    """Run the Mistral3 text encoder and build the 15360-dim conditioning sequence.

    Mirrors Flux2Pipeline._get_mistral_3_small_prompt_embeds: stack the hidden
    states from layers (10, 20, 30) and fold them into the channel dimension.
    """

    def __init__(self, text_encoder, out_layers=TEXT_ENCODER_OUT_LAYERS):
        super().__init__()
        self.text_encoder = text_encoder
        self.out_layers = tuple(out_layers)

    def forward(self, input_ids, attention_mask):
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack(
            [output.hidden_states[k] for k in self.out_layers], dim=1
        )  # (B, n_layers, seq, hidden)
        b, n, seq, hidden = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(b, seq, n * hidden)
        return prompt_embeds


class Flux2VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKLFlux2 as (z) -> image tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count. Sharding happens on the "model"
# axis; "batch" is size 1 for inference here.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")

# Convention (matches hidream_i1 / krea_realtime_video):
#   column-parallel weight (split output features) -> ("model", "batch")
#   row-parallel weight    (split input features)  -> ("batch", "model")
_COL = ("model", "batch")
_ROW = ("batch", "model")


def shard_flux2_transformer_specs(transformer) -> dict:
    """Tensor-parallel shard specs for Flux2Transformer2DModel.

    Attention: column-parallel Q/K/V (and added text-stream projections),
    row-parallel output. FeedForward: column-parallel linear_in (SwiGLU
    gate+up), row-parallel linear_out. Single-stream blocks fuse QKV + MLP in
    one column-parallel matmul and a single row-parallel output. Norms,
    modulation, and the tiny embed/proj layers are left replicated.
    """
    specs = {}

    def _shard_double_attn(attn):
        # Image stream Q/K/V/O (bias=False throughout the FLUX.2 DiT).
        specs[attn.to_q.weight] = _COL
        specs[attn.to_k.weight] = _COL
        specs[attn.to_v.weight] = _COL
        specs[attn.to_out[0].weight] = _ROW
        # Text-stream added projections.
        specs[attn.add_q_proj.weight] = _COL
        specs[attn.add_k_proj.weight] = _COL
        specs[attn.add_v_proj.weight] = _COL
        specs[attn.to_add_out.weight] = _ROW

    def _shard_ff(ff):
        # Flux2FeedForward: linear_in (dim -> inner*2, SwiGLU), linear_out (inner -> dim).
        specs[ff.linear_in.weight] = _COL
        specs[ff.linear_out.weight] = _ROW

    # Dual-stream blocks.
    for block in transformer.transformer_blocks:
        _shard_double_attn(block.attn)
        _shard_ff(block.ff)
        _shard_ff(block.ff_context)

    # Single-stream blocks: fused QKV+MLP projection (column) and output (row).
    for block in transformer.single_transformer_blocks:
        specs[block.attn.to_qkv_mlp_proj.weight] = _COL
        specs[block.attn.to_out.weight] = _ROW

    return specs


def shard_mistral3_specs(text_encoder) -> dict:
    """Tensor-parallel shard specs for the Mistral3 language model.

    Column-parallel q/k/v/gate/up, row-parallel o/down. The vision tower and
    multi-modal projector are not exercised by text-only prompts and are left
    replicated. Embedding and norms are left replicated.
    """
    specs = {}
    lm = text_encoder.model.language_model  # MistralModel

    for layer in lm.layers:
        attn = layer.self_attn
        specs[attn.q_proj.weight] = _COL
        specs[attn.k_proj.weight] = _COL
        specs[attn.v_proj.weight] = _COL
        specs[attn.o_proj.weight] = _ROW

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = _COL
        specs[mlp.up_proj.weight] = _COL
        specs[mlp.down_proj.weight] = _ROW

    return specs
