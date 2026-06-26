# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Component loaders and wrappers for black-forest-labs/FLUX.1-dev (FluxPipeline).

Components (each loadable / testable in isolation, mirroring the FLUX.2 bringup):
  - ClipTextEncoder → CLIPTextModel        (pooled text embedding)
  - T5TextEncoder   → T5EncoderModel       (sequence text embedding)
  - Transformer     → FluxTransformer2DModel (single denoise step)
  - Vae             → AutoencoderKL decoder

Bringup resolution: 1024x1024 pixels (FLUX.1-dev native output resolution).
"""

import torch

REPO_ID = "black-forest-labs/FLUX.1-dev"
DTYPE = torch.bfloat16

# FluxPipeline defaults aligned with diffusers/pipelines/flux/pipeline_flux.py
PROMPT = "An astronaut riding a horse in a futuristic city"
HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 3.5  # FLUX.1-dev (Schnell uses 0.0); Dev is guidance-distilled
MAX_SEQUENCE_LENGTH = 512
SEED = 42

VAE_SCALE_FACTOR = 8
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2  # 16

# Transformer config (FLUX.1-dev): in_channels = 64 -> 16 latent channels, 64 packed.
TRANSFORMER_IN_CHANNELS = 64
NUM_LATENT_CHANNELS = TRANSFORMER_IN_CHANNELS // 4  # 16
PACKED_LATENT_C = TRANSFORMER_IN_CHANNELS  # 64 (num_latent_channels * 4)

# Conditioning dims feeding the transformer (FLUX.1-dev):
#   pooled_projections: CLIP-L pooled    -> 768
#   encoder_hidden_states: T5-XXL hidden -> 4096
POOLED_PROJECTION_DIM = 768
JOINT_ATTENTION_DIM = 4096


def _latent_grid_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    """Packed-latent grid (h, w) at the given output resolution.

    Mirrors FluxPipeline.prepare_latents / _unpack_latents geometry.
    """
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return h // 2, w // 2


LATENT_GRID_H, LATENT_GRID_W = _latent_grid_hw()
LATENT_PACKED_SEQ = LATENT_GRID_H * LATENT_GRID_W


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_clip_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import CLIPTextModel

    return CLIPTextModel.from_pretrained(
        REPO_ID, subfolder="text_encoder", torch_dtype=dtype
    ).eval()


def load_t5_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID, subfolder="text_encoder_2", torch_dtype=dtype
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(
        REPO_ID, subfolder="transformer", torch_dtype=dtype
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        REPO_ID, subfolder="vae", torch_dtype=dtype
    ).eval()


def load_clip_tokenizer():
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


def load_t5_tokenizer():
    from transformers import T5TokenizerFast

    return T5TokenizerFast.from_pretrained(REPO_ID, subfolder="tokenizer_2")


# ---------------------------------------------------------------------------
# Tokenization (mirrors FluxPipeline._get_{clip,t5}_prompt_embeds)
# ---------------------------------------------------------------------------


def tokenize_clip(prompt: str = PROMPT) -> torch.Tensor:
    tokenizer = load_clip_tokenizer()
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def tokenize_t5(
    prompt: str = PROMPT, *, max_sequence_length: int = MAX_SEQUENCE_LENGTH
) -> torch.Tensor:
    tokenizer = load_t5_tokenizer()
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    return text_inputs.input_ids


# ---------------------------------------------------------------------------
# Component wrappers (tensor-in / tensor-out)
# ---------------------------------------------------------------------------


class ClipTextEncoderWrapper(torch.nn.Module):
    """Match FluxPipeline._get_clip_prompt_embeds (returns pooled embedding)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, output_hidden_states=False).pooler_output


class T5TextEncoderWrapper(torch.nn.Module):
    """Match FluxPipeline._get_t5_prompt_embeds (returns sequence embedding)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, output_hidden_states=False)[0]


class FluxTransformerWrapper(torch.nn.Module):
    """Single denoise step: same signature as FluxTransformer2DModel.forward."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]


def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Port of FluxPipeline._unpack_latents (pipeline_flux.py:529)."""
    batch_size, num_patches, channels = latents.shape
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, h, w)


class FluxVAEDecoderWrapper(torch.nn.Module):
    """Decode path after unpack + denorm (FluxPipeline.__call__ lines 1006-1008)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = _unpack_latents(latents, HEIGHT, WIDTH)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Packed latents [1, (h//2)*(w//2), 64] (FluxPipeline.prepare_latents layout)."""
    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(
        1,
        NUM_LATENT_CHANNELS,
        LATENT_GRID_H,
        2,
        LATENT_GRID_W,
        2,
        dtype=dtype,
        generator=gen,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(1, LATENT_PACKED_SEQ, PACKED_LATENT_C)


def prepare_text_ids(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(seq_len, 3, dtype=dtype)


def prepare_latent_image_ids(
    height: int, width: int, dtype: torch.dtype
) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    return latent_image_ids.reshape(-1, 3).to(dtype=dtype)


def make_pooled_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Synthetic CLIP pooled embedding [1, 768] for the Transformer test.

    Uses random conditioning rather than running the real CLIP encoder: the
    transformer graph compiles/runs identically and is compared against its own
    CPU reference on the same tensors, so real text features add nothing but
    would load the encoder into host RAM. Loading the real ~9.5 GB T5 (for the
    sequence embeds) alongside the ~23.8 GB transformer overflowed the host
    memory cgroup (OOM-kill), so both conditioning tensors are synthetic.
    """
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(1, POOLED_PROJECTION_DIM, dtype=dtype, generator=gen)


def make_prompt_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Synthetic T5 sequence embedding [1, MAX_SEQUENCE_LENGTH, 4096]."""
    gen = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(
        1, MAX_SEQUENCE_LENGTH, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen
    )


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Packed latents [1, seq, 64] feeding the VAE decoder component."""
    return make_packed_latents(dtype)
