# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen/Qwen-Image (QwenImagePipeline).

The Qwen-Image text-to-image pipeline is a multi-stage diffusion model that is
brought up component-by-component (the denoising loop / scheduler stay in host
Python). Components:

  - TextEncoder  -> Qwen2_5_VLForConditionalGeneration (~8.3B, Qwen2.5-VL-7B LM)
  - Transformer  -> QwenImageTransformer2DModel        (~20B MMDiT, 60 blocks)
  - Vae          -> AutoencoderKLQwenImage decoder      (~0.2B 3D causal VAE)

Native generation resolution is 1024x1024 (default_sample_size=128 *
vae_scale_factor=8). All component sample inputs are sized for that resolution.
"""

import torch

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16

# QwenImagePipeline defaults (diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
PROMPT = (
    "A coffee shop entrance features a chalkboard sign reading "
    '"Qwen Coffee, $2 per cup," with a neon light beside it displaying a '
    "cup of coffee. Photorealistic, warm morning light."
)
NEGATIVE_PROMPT = " "

# Native / maximum supported generation resolution.
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 42

# Prompt-encoding template (drops the 34-token system prefix after encoding).
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024

# VAE: 2 ** len(temperal_downsample) == 8 spatial compression.
VAE_SCALE_FACTOR = 8
# Pixel dims must be divisible by vae_scale_factor * 2 (8x compression + 2x2 patch).
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2  # 16

# Transformer config (QwenImageTransformer2DModel @ Qwen/Qwen-Image)
TRANSFORMER_IN_CHANNELS = 64
NUM_LATENT_CHANNELS = TRANSFORMER_IN_CHANNELS // 4  # 16 (== vae z_dim)
JOINT_ATTENTION_DIM = 3584  # text-encoder hidden size
PATCH_SIZE = 2
VAE_Z_DIM = 16

# Synthetic text-conditioning sequence length for the standalone transformer test
# (a real prompt after template trimming is typically a few dozen tokens).
TXT_SEQ_LEN = 64


def _latent_grid_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    """Packed-latent grid (number of 2x2 patches along H and W)."""
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return h // 2, w // 2


# Packed-latent geometry at the native resolution.
LATENT_GRID_H, LATENT_GRID_W = _latent_grid_hw()          # 64, 64
LATENT_PACKED_SEQ = LATENT_GRID_H * LATENT_GRID_W          # 4096
# Unpacked VAE latent spatial dims.
VAE_LATENT_H = 2 * (HEIGHT // VAE_SPATIAL_ALIGN)           # 128
VAE_LATENT_W = 2 * (WIDTH // VAE_SPATIAL_ALIGN)            # 128

# img_shapes entry consumed by the transformer's RoPE builder: (frames, H, W).
IMG_SHAPES = [[(1, HEIGHT // VAE_SCALE_FACTOR // 2, WIDTH // VAE_SCALE_FACTOR // 2)]]

# (batch, model) mesh shapes by device count. The ~20B transformer is the
# sharding target; the text encoder shards too, the VAE fits on one chip.
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
        device_map="cpu",
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
# Tokenization / prompt embedding (mirrors QwenImagePipeline._get_qwen_prompt_embeds)
# ---------------------------------------------------------------------------
def tokenize_prompt(prompt: str = PROMPT):
    """Tokenize a prompt with the Qwen-Image system template applied."""
    tokenizer = load_tokenizer()
    drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
    txt = [PROMPT_TEMPLATE_ENCODE.format(prompt)]
    txt_tokens = tokenizer(
        txt,
        max_length=TOKENIZER_MAX_LENGTH + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return txt_tokens.input_ids, txt_tokens.attention_mask


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def encode_prompt_embeds(text_encoder, input_ids, attention_mask, dtype=DTYPE):
    """Run the text encoder and post-process exactly like the pipeline.

    Returns (prompt_embeds [B, seq, 3584], prompt_embeds_mask [B, seq]).
    """
    drop_idx = PROMPT_TEMPLATE_ENCODE_START_IDX
    with torch.no_grad():
        out = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = out.hidden_states[-1]
    split = _extract_masked_hidden(hidden_states, attention_mask)
    split = [e[drop_idx:] for e in split]
    attn_list = [torch.ones(e.size(0), dtype=torch.long) for e in split]
    max_seq = max(e.size(0) for e in split)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq - u.size(0), u.size(1))]) for u in split]
    )
    prompt_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq - u.size(0))]) for u in attn_list]
    )
    return prompt_embeds.to(dtype=dtype), prompt_mask


# ---------------------------------------------------------------------------
# Latent packing (mirrors QwenImagePipeline._pack_latents / _unpack_latents)
# ---------------------------------------------------------------------------
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )


def unpack_latents(latents, height, width, vae_scale_factor=VAE_SCALE_FACTOR):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, height, width)


# ---------------------------------------------------------------------------
# Wrapper modules — tensor-in / tensor-out forwards for compilation
# ---------------------------------------------------------------------------
class QwenTextEncoderWrapper(torch.nn.Module):
    """Qwen2.5-VL text encoder -> last hidden state [B, seq, 3584]."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return out.hidden_states[-1]


class QwenImageTransformerWrapper(torch.nn.Module):
    """Single denoise step of QwenImageTransformer2DModel.

    img_shapes / txt_seq_lens are static (resolution-derived) and baked in so the
    forward takes only tensors.
    """

    def __init__(self, transformer, img_shapes=None, txt_seq_lens=None):
        super().__init__()
        self.transformer = transformer
        self.img_shapes = img_shapes if img_shapes is not None else IMG_SHAPES
        self.txt_seq_lens = (
            txt_seq_lens if txt_seq_lens is not None else [TXT_SEQ_LEN]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
    ):
        # txt_seq_lens is deprecated (>=0.39.0); the mask carries the same info.
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=self.img_shapes,
            guidance=None,
            attention_kwargs={},
            return_dict=False,
        )[0]


class QwenVAEDecoderWrapper(torch.nn.Module):
    """Denormalize + decode unpacked latents -> image [B, 3, H, W]."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor):
        z_dim = self.vae.config.z_dim
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0]
        return image[:, :, 0]


# ---------------------------------------------------------------------------
# Sample-input makers (native 1024x1024 geometry)
# ---------------------------------------------------------------------------
def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Packed noise latents [1, 4096, 64]."""
    gen = torch.Generator().manual_seed(SEED)
    latents = torch.randn(
        1, 1, NUM_LATENT_CHANNELS, VAE_LATENT_H, VAE_LATENT_W,
        dtype=dtype, generator=gen,
    )
    return pack_latents(latents, 1, NUM_LATENT_CHANNELS, VAE_LATENT_H, VAE_LATENT_W)


def make_prompt_embeds(dtype: torch.dtype = DTYPE):
    """Synthetic text-conditioning [1, TXT_SEQ_LEN, 3584] + all-ones mask."""
    gen = torch.Generator().manual_seed(SEED)
    embeds = torch.randn(1, TXT_SEQ_LEN, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen)
    mask = torch.ones(1, TXT_SEQ_LEN, dtype=torch.int64)
    return embeds, mask


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Unpacked latent grid [1, 16, 1, 128, 128] (pre-denorm)."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, VAE_Z_DIM, 1, VAE_LATENT_H, VAE_LATENT_W, dtype=dtype, generator=gen
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron column/row parallel on the MMDiT)
# ---------------------------------------------------------------------------
def _add(specs, param, spec):
    if param is not None:
        specs[param] = spec


def shard_transformer_specs(transformer) -> dict:
    """Column/row-parallel shard spec for QwenImageTransformer2DModel."""
    specs = {}
    _add(specs, transformer.img_in.weight, ("model", None))
    _add(specs, transformer.img_in.bias, ("model",))
    _add(specs, transformer.txt_in.weight, ("model", None))
    _add(specs, transformer.txt_in.bias, ("model",))

    for block in transformer.transformer_blocks:
        attn = block.attn
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            proj = getattr(attn, name, None)
            if proj is not None:
                _add(specs, proj.weight, ("model", None))
                _add(specs, proj.bias, ("model",))
        # output projections: row-parallel
        if hasattr(attn, "to_out") and len(attn.to_out) > 0:
            _add(specs, attn.to_out[0].weight, (None, "model"))
            _add(specs, attn.to_out[0].bias, (None,))
        if getattr(attn, "to_add_out", None) is not None:
            _add(specs, attn.to_add_out.weight, (None, "model"))
            _add(specs, attn.to_add_out.bias, (None,))

        # FeedForward: net[0].proj (in, column) -> net[2] (out, row)
        for mlp in (block.img_mlp, block.txt_mlp):
            _add(specs, mlp.net[0].proj.weight, ("model", None))
            _add(specs, mlp.net[0].proj.bias, ("model",))
            _add(specs, mlp.net[2].weight, (None, "model"))
            _add(specs, mlp.net[2].bias, (None,))

    _add(specs, transformer.proj_out.weight, (None, None))
    return specs


def shard_text_encoder_specs(text_encoder) -> dict:
    """Column/row-parallel shard spec for the Qwen2.5-VL language tower."""
    specs = {}
    lm = text_encoder
    for attr in ("model", "language_model", "model"):
        if hasattr(lm, attr):
            lm = getattr(lm, attr)
    layers = getattr(lm, "layers", None)
    if layers is None:
        return specs
    for layer in layers:
        sa = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, name)
            _add(specs, proj.weight, ("model", None))
            _add(specs, proj.bias, ("model",))
        _add(specs, sa.o_proj.weight, (None, "model"))
        mlp = layer.mlp
        _add(specs, mlp.gate_proj.weight, ("model", None))
        _add(specs, mlp.up_proj.weight, ("model", None))
        _add(specs, mlp.down_proj.weight, (None, "model"))
    return specs
