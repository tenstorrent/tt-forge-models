# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen/Qwen-Image (QwenImagePipeline).

Components:
  - TextEncoder  -> Qwen2_5_VLForConditionalGeneration  (~8.3B, text path only)
  - Transformer  -> QwenImageTransformer2DModel         (~20B MMDiT denoiser)
  - Vae          -> AutoencoderKLQwenImage decoder       (~0.25B, 3D video VAE)

Bringup resolution: 1328x1328 pixels (Qwen-Image native 1:1 resolution from the
model card; QwenImagePipeline.__call__ default_sample_size*vae_scale_factor=1024
is the diffusers fallback, but the model is trained/recommended at 1328).
"""

import torch

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16

# QwenImagePipeline defaults (diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
PROMPT = (
    "A coffee shop entrance features a chalkboard sign reading "
    '"Qwen Coffee, $2 per cup," with a neon light beside it displaying a '
    "vibrant cup of coffee. Photorealistic, warm morning light."
)
HEIGHT = 1328
WIDTH = 1328
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42

# Prompt template and template-prefix drop (pipeline _get_qwen_prompt_embeds).
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024

# VAE geometry: vae_scale_factor = 2 ** len(temperal_downsample) = 8.
VAE_SCALE_FACTOR = 8
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2  # 16
VAE_Z_DIM = 16  # AutoencoderKLQwenImage z_dim

# Transformer config (QwenImageTransformer2DModel).
TRANSFORMER_IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584
NUM_CHANNELS_LATENTS = TRANSFORMER_IN_CHANNELS // 4  # 16


def _latent_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    """Unpacked latent spatial size (pipeline.prepare_latents)."""
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return h, w


LATENT_H, LATENT_W = _latent_hw()  # 166, 166 at 1328
# Packed (patchified) image sequence into the transformer.
LATENT_PACKED_SEQ = (LATENT_H // 2) * (LATENT_W // 2)  # 83*83 = 6889
# img_shapes entry for RoPE: (frames, packed_h, packed_w).
IMG_SHAPE = (1, LATENT_H // 2, LATENT_W // 2)  # (1, 83, 83)

# Representative encoded text length for the standalone transformer device test
# (the composite pipeline feeds the real encoder output instead).
TEXT_SEQ_LEN = 256

# (batch, model) mesh shapes by device count. Weights shard along "model"; every
# device sits on that axis so the ~40 GB bf16 transformer / ~16 GB text encoder
# shard across all chips instead of replicating (see flux2 model_utils note).
MESH_SHAPES = {32: (1, 32), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------
def load_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_tokenizer():
    from transformers import Qwen2Tokenizer

    return Qwen2Tokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKLQwenImage

    return AutoencoderKLQwenImage.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


def tokenize_prompt(prompt: str = PROMPT):
    """Tokenize a prompt with the pipeline's chat template (single sample)."""
    tokenizer = load_tokenizer()
    txt = [PROMPT_TEMPLATE_ENCODE.format(prompt)]
    txt_tokens = tokenizer(
        txt,
        max_length=TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_ENCODE_START_IDX,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return txt_tokens.input_ids, txt_tokens.attention_mask


# ---------------------------------------------------------------------------
# Component wrappers (tensor-in / tensor-out, single forward)
# ---------------------------------------------------------------------------
class QwenTextEncoderWrapper(torch.nn.Module):
    """Match QwenImagePipeline._get_qwen_prompt_embeds: return last hidden state.

    The pipeline then drops the template prefix and pads; that host-side glue is
    applied in the composite, not on device. The device-validated op is the
    Qwen2.5-VL text forward producing hidden_states[-1] of shape
    (batch, seq, hidden_size=3584).
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return output.hidden_states[-1]


class QwenImageTransformerWrapper(torch.nn.Module):
    """Single denoise step: QwenImageTransformer2DModel.forward (guidance_embeds=False)."""

    def __init__(self, transformer, img_shapes):
        super().__init__()
        self.transformer = transformer
        # img_shapes is host-side metadata (list of tuples) for RoPE, not a tensor.
        self.img_shapes = img_shapes

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=self.img_shapes,
            guidance=None,
            return_dict=False,
        )[0]


class QwenImageVAEDecoderWrapper(torch.nn.Module):
    """Decode normalized latents -> image (QwenImagePipeline.__call__ decode path).

    Input is the already unpacked + denormalized latent of shape
    (B, z_dim, 1, H_lat, W_lat); output is the RGB image (B, 3, H, W).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]


# ---------------------------------------------------------------------------
# Synthetic inputs for the standalone component device tests
# ---------------------------------------------------------------------------
def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_PACKED_SEQ, TRANSFORMER_IN_CHANNELS, dtype=dtype, generator=gen
    )


def make_prompt_embeds(dtype: torch.dtype = DTYPE, seq_len: int = TEXT_SEQ_LEN):
    """Synthetic conditioning (embeds, mask) for the transformer device test."""
    gen = torch.Generator().manual_seed(SEED)
    embeds = torch.randn(1, seq_len, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen)
    mask = torch.ones(1, seq_len, dtype=torch.int64)
    return embeds, mask


def make_vae_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Normalized latent grid (1, z_dim, 1, H_lat, W_lat) for the VAE decode test."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, VAE_Z_DIM, 1, LATENT_H, LATENT_W, dtype=dtype, generator=gen
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron column -> row)
# ---------------------------------------------------------------------------
def _add(specs: dict, param, spec) -> None:
    """Register a partition spec only for real parameters (skip None)."""
    if param is not None:
        specs[param] = spec


def _resolve_language_model(encoder):
    """Descend wrapper modules to the decoder stack that owns `.layers`.

    Qwen2_5_VLForConditionalGeneration nests the language decoder as
    ``encoder.model.language_model`` (a Qwen2_5_VLModel wraps a vision tower +
    language model). Keep unwrapping common attributes until reaching the module
    exposing ``.layers``.
    """
    module = encoder
    visited = {id(module)}
    while not hasattr(module, "layers"):
        for attr in ("language_model", "model", "text_model"):
            inner = getattr(module, attr, None)
            if inner is not None and id(inner) not in visited:
                module = inner
                visited.add(id(module))
                break
        else:
            break
    return module


def shard_text_encoder_specs(encoder) -> dict:
    """Megatron column/row shard for the Qwen2.5-VL language-model blocks.

    num_attention_heads=28 and num_key_value_heads=4 both divide 4 (the qb2
    model axis), so q/k/v column-shard cleanly across 4 chips.
    """
    specs = {}
    language_model = _resolve_language_model(encoder)

    if hasattr(language_model, "embed_tokens"):
        _add(specs, language_model.embed_tokens.weight, (None, None))

    layers = getattr(language_model, "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name)
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))
        _add(specs, sa.o_proj.weight, (None, "model"))
        _add(specs, getattr(sa.o_proj, "bias", None), (None,))

        mlp = layer.mlp
        _add(specs, mlp.gate_proj.weight, ("model", None))
        _add(specs, mlp.up_proj.weight, ("model", None))
        _add(specs, mlp.down_proj.weight, (None, "model"))

        _add(specs, layer.input_layernorm.weight, (None,))
        _add(specs, layer.post_attention_layernorm.weight, (None,))

    if hasattr(language_model, "norm"):
        _add(specs, language_model.norm.weight, (None,))

    return specs


def _shard_qwenimage_attention(attn, specs: dict) -> None:
    """QwenDoubleStreamAttnProcessor attention (joint image+text streams)."""
    # Image-stream and text-stream QKV projections -> column parallel.
    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, proj_name, None)
        if proj is not None:
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))

    # Output projections -> row parallel.
    to_out = getattr(attn, "to_out", None)
    if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0:
        _add(specs, to_out[0].weight, (None, "model"))
        _add(specs, getattr(to_out[0], "bias", None), (None,))
    if getattr(attn, "to_add_out", None) is not None:
        _add(specs, attn.to_add_out.weight, (None, "model"))
        _add(specs, getattr(attn.to_add_out, "bias", None), (None,))

    # qk RMSNorms are per-head-dim; replicate.
    for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add(specs, getattr(norm, "weight", None), ("model",))


def _shard_feed_forward(ff, specs: dict) -> None:
    """diffusers FeedForward: net[0].proj (column) -> net[2] (row)."""
    net = ff.net
    _add(specs, net[0].proj.weight, ("model", None))
    _add(specs, getattr(net[0].proj, "bias", None), ("model",))
    _add(specs, net[2].weight, (None, "model"))
    _add(specs, getattr(net[2], "bias", None), (None,))


def shard_transformer_specs(transformer) -> dict:
    """Megatron column/row shard for QwenImageTransformer2DModel.

    num_attention_heads=24 divides 4. Input/output projections, modulations and
    norms are replicated (small, applied elementwise); attention QKV/out and the
    img/txt MLPs are the column->row sharded pairs (one all-reduce each).
    """
    specs = {}

    # Input projections + text norm: replicate.
    _add(specs, transformer.img_in.weight, (None, None))
    _add(specs, getattr(transformer.img_in, "bias", None), (None,))
    _add(specs, transformer.txt_in.weight, (None, None))
    _add(specs, getattr(transformer.txt_in, "bias", None), (None,))
    _add(specs, getattr(transformer.txt_norm, "weight", None), (None,))

    for block in transformer.transformer_blocks:
        _shard_qwenimage_attention(block.attn, specs)
        _shard_feed_forward(block.img_mlp, specs)
        _shard_feed_forward(block.txt_mlp, specs)
        # Modulation linears produce elementwise shift/scale/gate -> replicate.
        _add(specs, block.img_mod[1].weight, (None, None))
        _add(specs, getattr(block.img_mod[1], "bias", None), (None,))
        _add(specs, block.txt_mod[1].weight, (None, None))
        _add(specs, getattr(block.txt_mod[1], "bias", None), (None,))

    # Output head: replicate.
    if hasattr(transformer.norm_out, "linear"):
        _add(specs, transformer.norm_out.linear.weight, (None, None))
        _add(specs, getattr(transformer.norm_out.linear, "bias", None), (None,))
    _add(specs, transformer.proj_out.weight, (None, None))
    _add(specs, getattr(transformer.proj_out, "bias", None), (None,))

    return specs
