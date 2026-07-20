# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen/Qwen-Image (QwenImagePipeline).

Components:
  - TextEncoder  → Qwen2_5_VLForConditionalGeneration  (~7B, text-only usage)
  - Transformer  → QwenImageTransformer2DModel (MMDiT)  (~20B)
  - Vae          → AutoencoderKLQwenImage decoder        (3D video-style VAE)

Output resolution is 1024x1024 (latent grid 64x64 → packed sequence length 4096).
"""

import torch

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16

# Inference config matching the official Qwen/Qwen-Image source
# (https://huggingface.co/Qwen/Qwen-Image, diffusers QwenImagePipeline).
PROMPT = "A red cube on a white table, studio lighting, sharp focus."
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."
NEGATIVE_PROMPT = ""
# Org-source 1:1 default resolution: 1328x1328 -> latent 166x166 -> 2x2 patch
# -> 83x83 = 6889 image tokens. This is the real inference activation size; the
# joint-attention activation scales with (img_seq + txt_seq)^2 so the ~20B
# transformer OOMs on a 2/4-chip mesh at this resolution (OOM to be investigated).
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

    Mirrors QwenImagePipeline._get_qwen_prompt_embeds: ``padding=True`` (pad to the
    longest sequence, i.e. no padding for a single prompt) rather than pad to
    ``max_length``. Padding to the full 1024-token window would leave ~1000 masked
    padding positions whose Qwen2.5-VL hidden states are numerically unstable and
    diverge badly TT-vs-CPU, tanking PCC even when the real tokens match.
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
            1.0
            / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1)
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
    return torch.randn(
        1, TEXT_SEQ_LEN, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen
    )


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

    The Linear output is chunk(6)'d into (shift, scale, gate) for norm1 and norm2
    and applied elementwise to the full hidden state. Column-sharding that output
    dim would split across a chunk boundary and hand each device the wrong
    modulation slice (silently wrong per layer), so keep it REPLICATED.
    """
    lin = mod[-1] if isinstance(mod, (torch.nn.Sequential, torch.nn.ModuleList)) else mod
    if isinstance(lin, torch.nn.Linear):
        _add_shard_spec(specs, lin.weight, (None, None))
        _add_shard_spec(specs, lin.bias, (None,))


def shard_transformer_specs(transformer) -> dict:
    """Tensor-parallel shard spec for QwenImageTransformer2DModel.

    Only each dual-stream block's internal attention and FFN weights are sharded,
    so every block consumes and produces a replicated tensor. Embedders
    (img_in/txt_in), txt_norm, time_text_embed, norm_out and proj_out are omitted
    → replicated by default. Modulation projections stay explicitly replicated.
    """
    specs = {}

    for block in transformer.transformer_blocks:
        _shard_modulation(block.img_mod, specs)
        _shard_modulation(block.txt_mod, specs)
        _shard_qwenimage_joint_attention(block.attn, specs)
        _shard_qwenimage_feed_forward(block.img_mlp, specs)
        _shard_qwenimage_feed_forward(block.txt_mlp, specs)

    return specs
