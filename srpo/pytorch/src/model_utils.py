# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utilities for the SRPO (Tencent) text-to-image pipeline.

SRPO is a FLUX.1-dev fine-tune: it ships ONLY the denoising transformer weights
(`diffusion_pytorch_model.safetensors`, ~47.6 GB fp32) and reuses every other
FLUX.1-dev component unchanged (CLIP text encoder, T5-XXL text encoder, VAE,
scheduler). The transformer is the heavy per-step compute and the bringup target;
the other components are loaded straight from FLUX.1-dev for the composite.

Components:
  - Transformer  -> FluxTransformer2DModel (FLUX.1-dev arch + SRPO weights, ~12B)
  - TextEncoder  -> CLIPTextModel        (FLUX.1-dev, ~123M)
  - TextEncoder2 -> T5EncoderModel        (FLUX.1-dev, ~4.7B)
  - Vae          -> AutoencoderKL decoder (FLUX.1-dev, ~84M)
"""

from typing import Optional

import torch

# --- repositories -----------------------------------------------------------
SRPO_REPO = "tencent/SRPO"  # transformer-only fine-tune
BASE_REPO = "black-forest-labs/FLUX.1-dev"  # CLIP, T5, VAE, transformer config
SRPO_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"

# --- defaults ---------------------------------------------------------------
DTYPE = torch.bfloat16

# Bringup resolution (matches the in-repo flux / flux2 loaders, which compile at
# 128x128). The native composite resolution per the SRPO model card is 1024x1024,
# 50 steps, guidance 3.5, T5 sequence length 512.
HEIGHT = 128
WIDTH = 128
MAX_SEQUENCE_LENGTH = 256
GUIDANCE_SCALE = 3.5

# --- FLUX.1-dev transformer architecture constants (from transformer/config.json)
NUM_ATTENTION_HEADS = 24
ATTENTION_HEAD_DIM = 128
HIDDEN_SIZE = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 3072
IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 4096  # T5 hidden size
POOLED_PROJECTION_DIM = 768  # CLIP pooled size
VAE_SCALE_FACTOR = 8  # FLUX VAE: 2 ** (len(block_out_channels) - 1)

# packed-latent geometry derived from HEIGHT/WIDTH (mirrors FluxPipeline)
_LATENT_H = 2 * (HEIGHT // (VAE_SCALE_FACTOR * 2))  # 16
_LATENT_W = 2 * (WIDTH // (VAE_SCALE_FACTOR * 2))  # 16
PACKED_SEQ_LEN = (_LATENT_H // 2) * (_LATENT_W // 2)  # 64
PACKED_CHANNELS = (IN_CHANNELS // 4) * 4  # 64

# --- multi-device mesh ------------------------------------------------------
# Megatron tensor-parallel weights shard along the "model" axis only; the
# "batch" axis is size 1 here (batch=1). Use the full chip count on the model
# axis so the ~24 GB bf16 transformer shards across all chips.
MESH_SHAPES = {32: (1, 32), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------
def load_transformer(dtype: torch.dtype = DTYPE):
    """Build a FLUX.1-dev FluxTransformer2DModel and load SRPO fine-tuned weights.

    SRPO ships no config.json, so the architecture comes from FLUX.1-dev's
    transformer config and only the state dict is swapped in.
    """
    from diffusers import FluxTransformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = FluxTransformer2DModel.load_config(BASE_REPO, subfolder="transformer")
    model = FluxTransformer2DModel.from_config(config)

    weights_path = hf_hub_download(SRPO_REPO, SRPO_WEIGHTS_FILE)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    return model.to(dtype).eval()


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """CLIP text encoder (FLUX.1-dev text_encoder)."""
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(
        BASE_REPO, subfolder="text_encoder", torch_dtype=dtype
    ).eval()


def load_text_encoder_2(dtype: torch.dtype = DTYPE):
    """T5-XXL text encoder (FLUX.1-dev text_encoder_2)."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        BASE_REPO, subfolder="text_encoder_2", torch_dtype=dtype
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """VAE (FLUX.1-dev AutoencoderKL)."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        BASE_REPO, subfolder="vae", torch_dtype=dtype
    ).eval()


# ---------------------------------------------------------------------------
# Wrappers (positional forward, return_dict=False) so torch.compile/tracing sees
# plain tensor in / tensor out.
# ---------------------------------------------------------------------------
class FluxTransformerWrapper(torch.nn.Module):
    """Wrap FluxTransformer2DModel for a positional, tensor-out forward."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]


class CLIPTextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids, output_hidden_states=False).pooler_output


class T5TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids, output_hidden_states=False)[0]


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Synthetic input builders (transformer gate uses random embeddings of the right
# shape so the heavy CLIP/T5 encoders are not required for the per-component
# bringup; PCC is measured CPU-vs-TT on the same synthetic inputs).
# ---------------------------------------------------------------------------
def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.randn(1, PACKED_SEQ_LEN, PACKED_CHANNELS, dtype=dtype)


def make_prompt_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.randn(1, MAX_SEQUENCE_LENGTH, JOINT_ATTENTION_DIM, dtype=dtype)


def make_pooled_projections(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.randn(1, POOLED_PROJECTION_DIM, dtype=dtype)


def prepare_latent_image_ids(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    h, w = _LATENT_H // 2, _LATENT_W // 2
    ids = torch.zeros(h, w, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(h)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w)[None, :]
    return ids.reshape(-1, 3).to(dtype=dtype)


def prepare_text_ids(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.zeros(MAX_SEQUENCE_LENGTH, 3, dtype=dtype)


# ---------------------------------------------------------------------------
# Tensor-parallel shard specs for the FLUX.1 transformer (Megatron column->row).
# partition_spec tuples: ("model", None) column-shards dim 0; (None, "model")
# row-shards dim 1; ("model",) shards a 1D bias; (None,) replicates.
# ---------------------------------------------------------------------------
def _add(specs, param, spec):
    if param is not None:
        specs[param] = spec


def _shard_double_block(block, specs):
    attn = block.attn
    # Image-stream QKV: column-shard
    for name in ("to_q", "to_k", "to_v"):
        proj = getattr(attn, name, None)
        if proj is not None:
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))
    # Text-stream QKV: column-shard
    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, name, None)
        if proj is not None:
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))
    # Output projections: row-shard
    if getattr(attn, "to_out", None) is not None:
        _add(specs, attn.to_out[0].weight, (None, "model"))
        _add(specs, getattr(attn.to_out[0], "bias", None), (None,))
    if getattr(attn, "to_add_out", None) is not None:
        _add(specs, attn.to_add_out.weight, (None, "model"))
        _add(specs, getattr(attn.to_add_out, "bias", None), (None,))
    # MLPs: net[0].proj column, net[2] row
    for ff_name in ("ff", "ff_context"):
        ff = getattr(block, ff_name, None)
        if ff is not None:
            _add(specs, ff.net[0].proj.weight, ("model", None))
            _add(specs, getattr(ff.net[0].proj, "bias", None), ("model",))
            _add(specs, ff.net[2].weight, (None, "model"))
            _add(specs, getattr(ff.net[2], "bias", None), (None,))


def _shard_single_block(block, specs):
    attn = block.attn
    for name in ("to_q", "to_k", "to_v"):
        proj = getattr(attn, name, None)
        if proj is not None:
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))
    # Fused MLP input proj: column-shard
    if getattr(block, "proj_mlp", None) is not None:
        _add(specs, block.proj_mlp.weight, ("model", None))
        _add(specs, getattr(block.proj_mlp, "bias", None), ("model",))
    # Fused output proj (takes [attn_out | mlp_act]): row-shard
    if getattr(block, "proj_out", None) is not None:
        _add(specs, block.proj_out.weight, (None, "model"))
        _add(specs, getattr(block.proj_out, "bias", None), (None,))


def shard_transformer_specs(transformer) -> dict:
    """Megatron tensor-parallel partition specs for FluxTransformer2DModel.

    Heads (24) must divide the mesh model-axis. Per-head qk-norm weights
    (size head_dim) and all layernorms stay replicated.
    """
    specs = {}
    for block in transformer.transformer_blocks:
        _shard_double_block(block, specs)
    for block in transformer.single_transformer_blocks:
        _shard_single_block(block, specs)
    return specs
