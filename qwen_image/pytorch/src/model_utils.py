# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen/Qwen-Image (QwenImagePipeline).

Components:
  - TextEncoder  → Qwen2_5_VLForConditionalGeneration  (~7B, text-only usage)
  - Transformer  → QwenImageTransformer2DModel (MMDiT)  (~20B)
  - Vae          → AutoencoderKLQwenImage decoder        (3D video-style VAE)
"""

import torch

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16

# Inference config from the official Qwen/Qwen-Image usage example
# (https://huggingface.co/Qwen/Qwen-Image): source example prompt + positive
# magic, 50 steps, true_cfg_scale 4.0, seed 42. Resolution is the source's 1:1
# aspect ratio (its default example uses 1664x928 16:9).
PROMPT = (
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee '
    '😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it '
    "hangs a poster showing a beautiful Chinese woman, and beneath the poster is "
    'written "π≈3.1415926-53589793-23846264-33832795-02384197".'
)
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."
NEGATIVE_PROMPT = " "
# 1:1 aspect: 1328x1328 -> latent 166x166 -> 2x2 patches -> 83x83 = 6889 tokens.
HEIGHT = 1328
WIDTH = 1328
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42

# Text encoder (Qwen2.5-VL used in text-only mode by QwenImagePipeline)
TOKENIZER_MAX_LENGTH = 1024
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34

# Synthetic text-condition length for the standalone transformer component test.
TEXT_SEQ_LEN = 512

# Transformer config (Qwen/Qwen-Image)
TRANSFORMER_IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584
# QwenImagePipeline feeds the transformer scheduler_timestep / 1000; use a
# mid-denoising step for the standalone transformer input.
DENOISE_TIMESTEP = 500.0
TIMESTEP_SCALE = 1000.0

# VAE geometry: AutoencoderKLQwenImage is a 3D (video) VAE with 8x spatial
# compression (2 ** len(temperal_downsample) == 8) and z_dim latent channels.
VAE_SCALE_FACTOR = 8
LATENT_CHANNELS = 16  # AutoencoderKLQwenImage z_dim

# Packed latent geometry (see QwenImagePipeline.prepare_latents / _unpack_latents):
# latent grid dim = 2 * (pixels // (vae_scale_factor * 2)).
def _latent_grid_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    h = 2 * (int(height) // (VAE_SCALE_FACTOR * 2))
    w = 2 * (int(width) // (VAE_SCALE_FACTOR * 2))
    return h, w


LATENT_GRID_H, LATENT_GRID_W = _latent_grid_hw()
# Packed sequence length fed to the transformer (grid / 2 per axis, 2x2 patches).
LATENT_PACKED_SEQ = (LATENT_GRID_H // 2) * (LATENT_GRID_W // 2)
# img_shapes for RoPE: [[(frames, packed_h, packed_w)]] per QwenImagePipeline.
IMG_SHAPES = [[(1, LATENT_GRID_H // 2, LATENT_GRID_W // 2)]]
# VAE decoder input latent grid (unpacked): (z_dim, 1, H // 8, W // 8).
VAE_LATENT_H = HEIGHT // VAE_SCALE_FACTOR
VAE_LATENT_W = WIDTH // VAE_SCALE_FACTOR

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
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
        device_map="cpu",
    ).eval()


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


def tokenize_prompt(
    prompt: str = PROMPT,
    *,
    max_sequence_length: int = TOKENIZER_MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with the QwenImagePipeline prompt template.

    Mirrors QwenImagePipeline._get_qwen_prompt_embeds: ``padding=True`` (pad to
    the longest sequence) rather than to ``max_length``.
    """
    tokenizer = load_tokenizer()
    txt = PROMPT_TEMPLATE_ENCODE.format(prompt)
    inputs = tokenizer(
        [txt],
        max_length=max_sequence_length + PROMPT_TEMPLATE_ENCODE_START_IDX,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids, inputs.attention_mask


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKLQwenImage

    return AutoencoderKLQwenImage.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Wrappers (tensor-in / tensor-out, single denoise step)
# ---------------------------------------------------------------------------


class Qwen25VLTextEncoderWrapper(torch.nn.Module):
    """Match QwenImagePipeline._get_qwen_prompt_embeds (returns hidden_states[-1])."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]


class QwenImageTransformerWrapper(torch.nn.Module):
    """Single denoise step of QwenImageTransformer2DModel.

    ``img_shapes`` is RoPE shape metadata (constant for a fixed resolution), so it
    is baked into the module rather than traced as an input; only the tensors that
    actually vary per step are forward() arguments.
    """

    def __init__(self, transformer, img_shapes=IMG_SHAPES):
        super().__init__()
        self.transformer = transformer
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
            guidance=None,  # Qwen-Image base is not guidance-distilled.
            return_dict=False,
        )[0]


class QwenImageVAEDecoderWrapper(torch.nn.Module):
    """Decode path after unpack + latents denorm (QwenImagePipeline.__call__)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z_dim = self.vae.config.z_dim
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1)
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        # decode -> (B, C, T, H, W); pipeline takes the first (only) frame.
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_PACKED_SEQ, TRANSFORMER_IN_CHANNELS, dtype=dtype, generator=gen
    )


def make_synthetic_prompt_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Synthetic encoder_hidden_states with the real (1, seq, joint_attention_dim)
    shape, so the standalone transformer test does not have to run the 7B text
    encoder just to build inputs."""
    gen = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(1, TEXT_SEQ_LEN, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen)


def make_prompt_embeds_mask() -> torch.Tensor:
    return torch.ones(1, TEXT_SEQ_LEN, dtype=torch.int64)


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Unpacked latent grid (1, z_dim, 1, H // 8, W // 8) — QwenImage VAE decode input."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1,
        LATENT_CHANNELS,
        1,
        VAE_LATENT_H,
        VAE_LATENT_W,
        dtype=dtype,
        generator=gen,
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications (transformer only — ~20B params, needs a multi-chip mesh)
# ---------------------------------------------------------------------------


def _add_shard_spec(specs: dict, param: torch.Tensor | None, spec: tuple) -> None:
    """Register a partition spec only for real parameters (skip None weights/biases)."""
    if param is not None:
        specs[param] = spec


def _shard_qwenimage_joint_attention(attn, specs: dict) -> None:
    """QwenDoubleStreamAttnProcessor2_0 attention in the dual-stream blocks.

    Megatron column→row: image + text QKV projections are column-parallel, the two
    output projections are row-parallel. Per-head RMSNorm over head_dim stays whole
    per device, so it is replicated.
    """
    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, proj_name, None)
        if proj is not None:
            _add_shard_spec(specs, proj.weight, ("model", None))
            _add_shard_spec(specs, proj.bias, ("model",))

    to_out = attn.to_out
    if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0:
        out_proj = to_out[0]
    elif isinstance(to_out, torch.nn.Linear):
        out_proj = to_out
    else:
        out_proj = None
    if out_proj is not None:
        _add_shard_spec(specs, out_proj.weight, (None, "model"))
        _add_shard_spec(specs, out_proj.bias, (None,))

    if getattr(attn, "to_add_out", None) is not None:
        _add_shard_spec(specs, attn.to_add_out.weight, (None, "model"))
        _add_shard_spec(specs, attn.to_add_out.bias, (None,))

    for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add_shard_spec(specs, getattr(norm, "weight", None), (None,))


def _shard_qwenimage_feed_forward(ff, specs: dict) -> None:
    """FeedForward(activation_fn="gelu-approximate"): plain (non-gated) GELU.

    net[0] is GELU whose ``.proj`` is Linear(dim, inner) → column-parallel;
    net[-1] is Linear(inner, dim_out) → row-parallel. GELU is not gated, so
    column-splitting the inner dim is safe (no chunk() straddling like GEGLU).
    """
    act = ff.net[0]
    if hasattr(act, "proj"):
        _add_shard_spec(specs, act.proj.weight, ("model", None))
        _add_shard_spec(specs, act.proj.bias, ("model",))
    out_proj = ff.net[-1]
    if isinstance(out_proj, torch.nn.Linear):
        _add_shard_spec(specs, out_proj.weight, (None, "model"))
        _add_shard_spec(specs, out_proj.bias, (None,))


def _shard_modulation(mod, specs: dict) -> None:
    """img_mod / txt_mod = Sequential(SiLU, Linear(dim, 6*dim)).

    Shard contraction-parallel (None, "model"): each device holds 1/N of the
    weight and an all-reduce restores the full 6*dim output before the chunk(6),
    so the result matches the replicated case at 1/N the weight memory. A column
    split would straddle the chunk(6) boundary and is unsafe.
    """
    lin = (
        mod[-1] if isinstance(mod, (torch.nn.Sequential, torch.nn.ModuleList)) else mod
    )
    if isinstance(lin, torch.nn.Linear):
        _add_shard_spec(specs, lin.weight, (None, "model"))
        _add_shard_spec(specs, lin.bias, (None,))


def shard_transformer_specs(transformer) -> dict:
    """Tensor-parallel shard spec for QwenImageTransformer2DModel.

    Each dual-stream block's attention, FFN and modulation weights are sharded,
    so every block consumes and produces a replicated tensor (attention/FFN via
    Megatron column→row, modulation via contraction-parallel all-reduce). The
    small embedders (img_in/txt_in), txt_norm, time_text_embed, norm_out and
    proj_out are omitted → replicated by default (each < 12M params).
    """
    specs = {}

    for block in transformer.transformer_blocks:
        _shard_modulation(block.img_mod, specs)
        _shard_modulation(block.txt_mod, specs)
        _shard_qwenimage_joint_attention(block.attn, specs)
        _shard_qwenimage_feed_forward(block.img_mlp, specs)
        _shard_qwenimage_feed_forward(block.txt_mlp, specs)

    return specs


def _resolve_language_model(encoder):
    """Descend wrapper modules to the Qwen2.5 decoder stack that owns ``.layers``.

    The text encoder is Qwen2_5_VLForConditionalGeneration; its decoder is at
    ``model.language_model``. Keep unwrapping ``model`` / ``language_model`` /
    ``text_model`` until a module exposing ``layers`` is reached.
    """
    module = encoder
    visited = {id(module)}
    while not hasattr(module, "layers"):
        for attr in ("model", "language_model", "text_model"):
            inner = getattr(module, attr, None)
            if inner is not None and id(inner) not in visited:
                module = inner
                visited.add(id(module))
                break
        else:
            raise ValueError("Could not locate the language-model decoder stack")
    return module


def shard_text_encoder_specs(encoder) -> dict:
    """Tensor-parallel shard spec for the Qwen2.5-VL text encoder.

    Megatron column→row on the decoder layers: q/k/v projections are
    column-parallel, o_proj is row-parallel; MLP gate/up are column-parallel and
    down is row-parallel; layernorms and embeddings stay replicated. GQA divides
    cleanly at degree 4 (28 q-heads → 7, 4 kv-heads → 1). Replicated the encoder
    is ~16.6 GB/chip, which cannot coexist with the transformer; sharding drops
    it to ~4 GB/chip so both fit (the vision tower is unused in text-only mode
    and left replicated).
    """
    specs = {}
    lm = _resolve_language_model(encoder)

    if hasattr(lm, "embed_tokens"):
        _add_shard_spec(specs, lm.embed_tokens.weight, (None, None))

    for layer in lm.layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name)
            _add_shard_spec(specs, proj.weight, ("model", None))
            _add_shard_spec(specs, proj.bias, ("model",))
        _add_shard_spec(specs, sa.o_proj.weight, (None, "model"))
        _add_shard_spec(specs, getattr(sa.o_proj, "bias", None), (None,))

        mlp = layer.mlp
        _add_shard_spec(specs, mlp.gate_proj.weight, ("model", None))
        _add_shard_spec(specs, mlp.up_proj.weight, ("model", None))
        _add_shard_spec(specs, mlp.down_proj.weight, (None, "model"))

        _add_shard_spec(specs, getattr(layer.input_layernorm, "weight", None), (None,))
        _add_shard_spec(
            specs, getattr(layer.post_attention_layernorm, "weight", None), (None,)
        )

    if hasattr(lm, "norm"):
        _add_shard_spec(specs, getattr(lm.norm, "weight", None), (None,))

    return specs
