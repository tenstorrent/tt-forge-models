# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen-Image (text-to-image MMDiT pipeline).

Model: Qwen/Qwen-Image
Components:
  - text_encoder: Qwen2.5-VL text decoder (Qwen2_5_VLForConditionalGeneration) (~8.3B)
  - transformer:  QwenImageTransformer2DModel (MMDiT, 60 layers)              (~20.4B)
  - vae:          AutoencoderKLQwenImage decoder (3D causal VAE, z_dim=16)    (~0.25B)

The diffusers pipeline is FlowMatchEuler text-to-image:
  packed latents -> transformer (per-step) -> _unpack -> VAE.decode.
Latent packing folds a 2x2 spatial patch into the channel dim, so the
transformer sees in_channels = 16 * 4 = 64 and a flat token sequence.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16  # HF distributes the weights in bf16

# ---------------------------------------------------------------------------
# Inference shape constants
# ---------------------------------------------------------------------------
# Native 1:1 resolution recommended by the Qwen-Image model card (best quality).
# vae_scale_factor = 2 ** len(temperal_downsample) = 2**3 = 8; latent packing
# additionally halves H/W (patch_size 2).

IMAGE_H = 1328
IMAGE_W = 1328
VAE_SCALE_FACTOR = 8

# VAE latent grid (un-packed), as produced by prepare_latents:
#   latent_h = 2 * (H // (vae_scale_factor * 2))
LATENT_H = 2 * (IMAGE_H // (VAE_SCALE_FACTOR * 2))  # 166
LATENT_W = 2 * (IMAGE_W // (VAE_SCALE_FACTOR * 2))  # 166
Z_DIM = 16  # vae.config.z_dim — unpacked latent channels

# Packed transformer sequence: (latent_h//2) * (latent_w//2) image tokens.
PACK = 2
IMG_SEQ_LEN = (LATENT_H // PACK) * (LATENT_W // PACK)  # 83 * 83 = 6889
TRANSFORMER_IN_CHANNELS = Z_DIM * PACK * PACK  # 16 * 4 = 64
TRANSFORMER_OUT_CHANNELS = Z_DIM  # 16

# img_shapes the pipeline feeds the transformer for RoPE:
#   [[(1, H//vae_scale_factor//2, W//vae_scale_factor//2)]]
IMG_SHAPE = (1, IMAGE_H // VAE_SCALE_FACTOR // PACK, IMAGE_W // VAE_SCALE_FACTOR // PACK)  # (1, 83, 83)

# Text conditioning (Qwen2.5-VL last hidden state). max_sequence_length=512 in the
# pipeline; a moderate prompt after the chat template / drop_idx lands well under it.
TEXT_EMBED_DIM = 3584  # text_encoder hidden_size / transformer.joint_attention_dim
TEXT_SEQ_LEN = 256  # representative conditioning length for the component test
QWEN_VOCAB_SIZE = 151936

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Load the Qwen2.5-VL text encoder from the text_encoder subfolder.

    The pipeline feeds only input_ids/attention_mask (no pixel_values), so the
    vision tower never runs. Return just `.language_model` to avoid uploading the
    unused visual tower (replicated on every chip -> DRAM pressure).
    """
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()
    return getattr(encoder, "language_model", encoder)


def load_transformer(dtype: torch.dtype = DTYPE):
    """Load QwenImageTransformer2DModel from the transformer subfolder."""
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """Load AutoencoderKLQwenImage from the vae subfolder."""
    from diffusers import AutoencoderKLQwenImage

    return AutoencoderKLQwenImage.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — tensor-in / tensor-out for the tt-xla graph runner.
# Non-tensor pipeline args (img_shapes, txt_seq_lens) are baked in at __init__.
# ---------------------------------------------------------------------------


class QwenImageTransformerWrapper(torch.nn.Module):
    """Simplify QwenImageTransformer2DModel to tensor-only inputs/outputs.

    guidance_embeds is False for Qwen-Image, so `guidance` is never passed.
    img_shapes / txt_seq_lens are constant per resolution and are captured here.
    """

    def __init__(self, transformer, img_shape=IMG_SHAPE, txt_seq_len=TEXT_SEQ_LEN):
        super().__init__()
        self.transformer = transformer
        self.img_shapes = [[tuple(img_shape)]]
        self.txt_seq_lens = [int(txt_seq_len)]

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_mask):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=self.img_shapes,
            txt_seq_lens=self.txt_seq_lens,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only AutoencoderKLQwenImage decoder as (z) -> tensor.

    The Qwen-Image VAE is a 3D causal VAE: decode expects a 5D latent
    [B, C, T, H, W]; for text-to-image T = 1. Returns the decoded sample.

    AutoencoderKLQwenImage._decode() calls clear_cache() / _count_conv3d(), which
    iterate model.modules() mid-forward and break dynamo tracing on device. For the
    single-frame (T=1) text-to-image case we call the decoder directly with
    feat_cache=None, which is dynamo-traceable. (This differs from the cached path
    only in the causal-temporal padding at the single frame boundary.)
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decoder(z, feat_cache=None)


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron column->row)
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for the Qwen2.5-VL text decoder.

    Column-parallel (q, k, v, gate, up): ("model", "batch")
    Row-parallel   (o, down):            ("batch", "model")
    """
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
    """Shard specs for QwenImageTransformer2DModel.

    Qwen-Image is a dual-stream MMDiT: each of the 60 `transformer_blocks`
    carries an image stream and a text stream, joined by one Attention module
    with separate img (to_q/k/v, to_out) and txt (add_q/k/v_proj, to_add_out)
    projections, plus per-stream FeedForward (img_mlp / txt_mlp) and AdaLN
    modulation Linears (img_mod / txt_mod).

    Column-parallel (Q, K, V, FFN up, AdaLN): ("model", "batch")
    Row-parallel   (O, FFN down):             ("batch", "model")
    """
    specs = {}

    def _col(linear):
        specs[linear.weight] = ("model", "batch")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("model",)

    def _row(linear):
        specs[linear.weight] = ("batch", "model")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("batch",)

    def _shard_attn(attn):
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            p = getattr(attn, name, None)
            if p is not None:
                _col(p)
        for name in ("to_out", "to_add_out"):
            out = getattr(attn, name, None)
            if out is None:
                continue
            target = out[0] if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList)) else out
            _row(target)

    def _shard_ff(ff):
        # diffusers FeedForward = net[ GEGLU(proj), Dropout, Linear ]
        if not hasattr(ff, "net"):
            return
        if hasattr(ff.net[0], "proj"):
            _col(ff.net[0].proj)
        _row(ff.net[2])

    for block in getattr(transformer, "transformer_blocks", []):
        # AdaLN modulation Linears
        for mod_name in ("img_mod", "txt_mod"):
            mod = getattr(block, mod_name, None)
            if mod is None:
                continue
            lin = mod[1] if isinstance(mod, (torch.nn.Sequential, torch.nn.ModuleList)) else getattr(mod, "linear", None)
            if isinstance(lin, torch.nn.Linear):
                _col(lin)
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        for ff_name in ("img_mlp", "txt_mlp", "ff", "ff_context"):
            ff = getattr(block, ff_name, None)
            if ff is not None:
                _shard_ff(ff)

    return specs
