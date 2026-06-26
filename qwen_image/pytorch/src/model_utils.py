# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, wrappers and SPMD shard specs for Qwen-Image.

Model: Qwen/Qwen-Image (QwenImagePipeline)
Components:
  - text_encoder: Qwen2.5-VL text encoder                       (8.29B)
  - transformer:  QwenImageTransformer2DModel (dual-stream MMDiT) (20.43B)
  - vae:          AutoencoderKLQwenImage (3D, video-capable)     (~0.14B)

The transformer (40.9 GB in bf16) does not fit a single 32 GB Blackhole chip,
so it is brought up tensor-parallel via the ``get_mesh_config`` /
``load_shard_spec`` contract (Megatron column->row). The text encoder is also
sharded; the VAE fits a single chip.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16  # native distribution dtype

# ---------------------------------------------------------------------------
# Inference shape constants — native 1328x1328 (model-card 1:1 resolution)
# ---------------------------------------------------------------------------

IMAGE_H = 1328
IMAGE_W = 1328
VAE_SCALE_FACTOR = 8  # 2 ** len(vae.temperal_downsample)
PATCH = 2  # transformer patch_size

LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 166
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 166
GRID_H = LATENT_H // PATCH  # 83
GRID_W = LATENT_W // PATCH  # 83
IMG_SEQ = GRID_H * GRID_W  # 6889

VAE_Z_DIM = 16  # vae.config.z_dim
NUM_CHANNELS_LATENTS = 16  # transformer.config.in_channels // 4
TRANSFORMER_IN_CHANNELS = 64  # transformer.config.in_channels
PACKED_HIDDEN_DIM = NUM_CHANNELS_LATENTS * (PATCH * PATCH)  # 64

TEXT_EMBED_DIM = 3584  # transformer.config.joint_attention_dim / Qwen2.5-VL hidden
TRANSFORMER_TEXT_SEQ = 128  # conditioning sequence length fed to the denoiser
QWEN_VOCAB_SIZE = 151936

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load QwenImageTransformer2DModel from the transformer subfolder."""
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        REPO_ID, subfolder="transformer", torch_dtype=dtype
    ).eval()


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load the Qwen2.5-VL text encoder.

    The pipeline feeds only input_ids/attention_mask (no pixel_values), so the
    vision tower never runs. Return just ``.language_model`` to avoid uploading
    the unused visual tower replicated on every chip.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPO_ID, subfolder="text_encoder", torch_dtype=dtype
    ).eval()
    return getattr(encoder, "language_model", encoder)


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLQwenImage from the vae subfolder."""
    from diffusers import AutoencoderKLQwenImage

    return AutoencoderKLQwenImage.from_pretrained(
        REPO_ID, subfolder="vae", torch_dtype=dtype
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — tensor-in / tensor-out for the tt-xla graph runner.
# ---------------------------------------------------------------------------


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLQwenImage decode as (latents) -> tensor.

    The stock ``vae.decode`` calls ``clear_cache()`` (which iterates
    ``model.modules()``) and threads a stateful ``feat_cache`` python list
    through the decoder — both break dynamo tracing. For a single still image
    (T=1) the streaming temporal cache is unnecessary, so we run the decoder on
    the full latent with ``feat_cache=None`` (the non-cached causal-conv path),
    which is numerically equivalent and fully traceable.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        x = self.vae.post_quant_conv(latents)
        out = self.vae.decoder(x, feat_cache=None)
        return torch.clamp(out, min=-1.0, max=1.0)


class QwenImageTransformerWrapper(torch.nn.Module):
    """Simplify QwenImageTransformer2DModel.forward to tensor-only I/O.

    img_shapes is a static (per-resolution) constant, guidance is None
    (config.guidance_embeds == False), so neither needs to be an input tensor.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.img_shapes = [[(1, GRID_H, GRID_W)]]

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=self.img_shapes,
            guidance=None,
            return_dict=False,
        )[0]


class QwenTextEncoderWrapper(torch.nn.Module):
    """Return the last hidden state, as QwenImagePipeline does."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron column->row)
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for the Qwen2.5-VL text encoder (column q/k/v/gate/up, row o/down)."""
    specs = {}
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
        for proj in ("q_proj", "k_proj", "v_proj"):
            p = getattr(sa, proj)
            specs[p.weight] = ("model", "batch")
            if p.bias is not None:
                specs[p.bias] = ("model",)
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


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for QwenImageTransformer2DModel (dual-stream MMDiT).

    Column-parallel (Q/K/V, add Q/K/V, FFN up, AdaLN modulation): ("model", "batch")
    Row-parallel   (O, add O, FFN down):                          ("batch", "model")
    """
    specs = {}

    # Input projections (img_in: 64->inner, txt_in: 3584->inner): column-parallel.
    for name in ("img_in", "txt_in"):
        lin = getattr(transformer, name, None)
        if lin is not None and hasattr(lin, "weight"):
            specs[lin.weight] = ("model", "batch")
            if getattr(lin, "bias", None) is not None:
                specs[lin.bias] = ("model",)

    def _shard_attn(attn):
        for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                specs[proj.weight] = ("model", "batch")
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)
        for proj_name in ("to_out", "to_add_out"):
            out = getattr(attn, proj_name, None)
            if out is None:
                continue
            target = (
                out[0]
                if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                else out
            )
            if hasattr(target, "weight"):
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

    def _shard_ff(ff):
        # diffusers FeedForward = Sequential(GELU(proj), Dropout, Linear)
        if hasattr(ff, "net"):
            if hasattr(ff.net[0], "proj"):
                specs[ff.net[0].proj.weight] = ("model", "batch")
                if ff.net[0].proj.bias is not None:
                    specs[ff.net[0].proj.bias] = ("model",)
            specs[ff.net[2].weight] = ("batch", "model")
            if ff.net[2].bias is not None:
                specs[ff.net[2].bias] = ("batch",)

    def _shard_mod(mod):
        # img_mod / txt_mod = Sequential(SiLU, Linear(inner -> 6*inner))
        if isinstance(mod, (torch.nn.Sequential, torch.nn.ModuleList)):
            for sub in mod:
                if hasattr(sub, "weight"):
                    specs[sub.weight] = ("model", "batch")
                    if getattr(sub, "bias", None) is not None:
                        specs[sub.bias] = ("model",)

    for block in getattr(transformer, "transformer_blocks", []):
        for mod_name in ("img_mod", "txt_mod"):
            if hasattr(block, mod_name):
                _shard_mod(getattr(block, mod_name))
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        for ff_name in ("img_mlp", "txt_mlp"):
            if hasattr(block, ff_name):
                _shard_ff(getattr(block, ff_name))

    # Final output projection (inner -> 64): keep replicated on a (1, N) mesh.
    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs
