# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Qwen/Qwen-Image (QwenImagePipeline).

Qwen-Image is an MMDiT text-to-image pipeline:
  - TextEncoder  -> Qwen2_5_VLForConditionalGeneration  (~8B, hidden 3584)
  - Transformer  -> QwenImageTransformer2DModel         (~20B, 60 dual-stream blocks)
  - Vae          -> AutoencoderKLQwenImage decoder       (~0.13B, z_dim 16)

Bringup resolution: 128x128 pixels. The model's native resolution is 1328x1328;
128x128 is used for per-component compile/run validation of the 20B denoiser within
the bringup time budget (mirrors the FLUX.2 sibling MMDiT loader). The composite step
documents this as a deviation. Latent geometry is derived from the chosen resolution.
"""

import torch

REPO_ID = "Qwen/Qwen-Image"
DTYPE = torch.bfloat16

# QwenImagePipeline defaults (diffusers/pipelines/qwenimage/pipeline_qwenimage.py)
PROMPT = (
    "A coffee shop entrance features a chalkboard sign reading 'Qwen Coffee', "
    "with a cat sitting beside it, warm morning light, photorealistic."
)
NEGATIVE_PROMPT = " "
HEIGHT = 128
WIDTH = 128
NATIVE_HEIGHT = 1328
NATIVE_WIDTH = 1328
NUM_INFERENCE_STEPS = 8
# Qwen-Image is not guidance-distilled (transformer.config.guidance_embeds == False):
# true-CFG would double the per-step transformer calls. Keep it disabled for the
# bringup composite so each step is a single denoiser call.
TRUE_CFG_SCALE = 1.0
SEED = 42
MAX_SEQUENCE_LENGTH = 512

# Prompt template applied by QwenImagePipeline._get_qwen_prompt_embeds.
PROMPT_TEMPLATE_ENCODE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34

# Transformer geometry (QwenImageTransformer2DModel config).
TRANSFORMER_IN_CHANNELS = 64
JOINT_ATTENTION_DIM = 3584
NUM_LATENT_CHANNELS = TRANSFORMER_IN_CHANNELS // 4  # 16

VAE_SCALE_FACTOR = 8
VAE_SPATIAL_ALIGN = VAE_SCALE_FACTOR * 2  # 16
VAE_Z_DIM = 16

# Synthetic text-sequence length for the standalone Transformer component test,
# so the 20B denoiser can be validated without first running the 8B text encoder.
SYNTHETIC_TXT_SEQ = 64


def _latent_grid_hw(height: int = HEIGHT, width: int = WIDTH) -> tuple[int, int]:
    """Packed-latent grid (per-side patch count) for the denoiser, per prepare_latents."""
    h = 2 * (int(height) // VAE_SPATIAL_ALIGN)
    w = 2 * (int(width) // VAE_SPATIAL_ALIGN)
    return h, w


# Unpacked latent grid (channels-last spatial dims fed to the VAE).
LATENT_H, LATENT_W = _latent_grid_hw()  # 16, 16 at 128x128
LATENT_PACKED_SEQ = (LATENT_H // 2) * (LATENT_W // 2)  # 64
# img_shapes drives RoPE: (frames, grid_h, grid_w)
IMG_SHAPES = [(1, HEIGHT // VAE_SCALE_FACTOR // 2, WIDTH // VAE_SCALE_FACTOR // 2)]

# (batch, model) mesh shapes by device count. Weights shard along "model"; put every
# device on "model" so the 20B transformer's weights shard across all chips (the bf16
# weights are ~40 GB and must shard to fit a single Blackhole chip's 32 GB DRAM).
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
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


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


def tokenize_prompt(
    prompt: str = PROMPT,
    *,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize with the QwenImage prompt template (matches the pipeline)."""
    tokenizer = load_tokenizer()
    txt = [PROMPT_TEMPLATE_ENCODE.format(prompt)]
    inputs = tokenizer(
        txt,
        max_length=max_sequence_length + PROMPT_TEMPLATE_ENCODE_START_IDX,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs["input_ids"], inputs["attention_mask"]


# ---------------------------------------------------------------------------
# Wrappers (tensor-in / tensor-out, single compilable forward each)
# ---------------------------------------------------------------------------


class QwenImageTextEncoderWrapper(torch.nn.Module):
    """Return the final hidden state of the Qwen2.5-VL text encoder.

    Matches QwenImagePipeline._get_qwen_prompt_embeds, which takes
    encoder_hidden_states.hidden_states[-1]. Template-token dropping / repacking
    is host glue applied after this forward.
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
    """Single denoise step of QwenImageTransformer2DModel.

    img_shapes is static RoPE geometry, so it is held on the module and not passed
    as a graph input. guidance is None (Qwen-Image is not guidance-distilled).
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
            guidance=None,
            attention_kwargs={},
            return_dict=False,
        )[0]


class QwenImageVAEDecoderWrapper(torch.nn.Module):
    """Denormalize latents and decode to an image (QwenImagePipeline.__call__ tail).

    Input is the unpacked latent grid (B, z_dim, T=1, H, W). The pipeline computes
    latents = latents / (1/std) + mean = latents * std + mean before decoding.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z_dim = self.vae.config.z_dim
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(device=latents.device, dtype=latents.dtype)
        )
        std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, z_dim, 1, 1, 1)
            .to(device=latents.device, dtype=latents.dtype)
        )
        latents = latents * std + mean
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]


# ---------------------------------------------------------------------------
# Synthetic inputs for standalone component tests
# ---------------------------------------------------------------------------


def make_packed_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, LATENT_PACKED_SEQ, TRANSFORMER_IN_CHANNELS, dtype=dtype, generator=gen
    )


def make_prompt_embeds(
    dtype: torch.dtype = DTYPE, txt_seq: int = SYNTHETIC_TXT_SEQ
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(1, txt_seq, JOINT_ATTENTION_DIM, dtype=dtype, generator=gen)


def make_prompt_embeds_mask(txt_seq: int = SYNTHETIC_TXT_SEQ) -> torch.Tensor:
    return torch.ones(1, txt_seq, dtype=torch.int64)


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Unpacked latent grid (1, z_dim, 1, LATENT_H, LATENT_W)."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(
        1, VAE_Z_DIM, 1, LATENT_H, LATENT_W, dtype=dtype, generator=gen
    )


# ---------------------------------------------------------------------------
# Latent (un)packing — host glue reused by the composite pipeline
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
# SPMD shard specifications (Megatron column->row, minimal CCL)
# ---------------------------------------------------------------------------


def _add(specs: dict, param, spec) -> None:
    """Register a partition spec only for real parameters (skip None weights/biases)."""
    if param is not None:
        specs[param] = spec


def _resolve_text_transformer(encoder):
    """Descend wrapper modules to the decoder stack that owns `.layers`.

    Qwen2_5_VLForConditionalGeneration nests the language model below
    ``.model``/``.language_model``; the decoder block stack (with ``.layers``)
    lives below those wrappers.
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
    """Shard the Qwen2.5-VL language-model decoder blocks (column/row parallel)."""
    specs = {}
    language_model = _resolve_text_transformer(encoder)

    if hasattr(language_model, "embed_tokens"):
        specs[language_model.embed_tokens.weight] = (None, None)

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
    """QwenImage joint attention: image stream (to_*) + text stream (add_*_proj)."""
    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, proj_name, None)
        if proj is not None:
            _add(specs, proj.weight, ("model", None))
            _add(specs, getattr(proj, "bias", None), ("model",))

    # to_out is a ModuleList [Linear, Dropout]; row-shard the linear.
    to_out = attn.to_out
    out_proj = to_out[0] if isinstance(to_out, torch.nn.ModuleList) else to_out
    _add(specs, out_proj.weight, (None, "model"))
    _add(specs, getattr(out_proj, "bias", None), (None,))

    if getattr(attn, "to_add_out", None) is not None:
        _add(specs, attn.to_add_out.weight, (None, "model"))
        _add(specs, getattr(attn.to_add_out, "bias", None), (None,))

    # Per-head RMSNorms (over head_dim) stay replicated.
    for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add(specs, getattr(norm, "weight", None), (None,))


def _shard_qwenimage_feed_forward(ff, specs: dict) -> None:
    """diffusers FeedForward: net[0] is GELU(.proj) (column), net[-1] is Linear (row)."""
    net = ff.net
    gelu = net[0]
    if hasattr(gelu, "proj"):
        _add(specs, gelu.proj.weight, ("model", None))
        _add(specs, getattr(gelu.proj, "bias", None), ("model",))
    out_linear = net[-1]
    _add(specs, out_linear.weight, (None, "model"))
    _add(specs, getattr(out_linear, "bias", None), (None,))


def shard_transformer_specs(transformer) -> dict:
    """Shard QwenImageTransformer2DModel (60 dual-stream MMDiT blocks)."""
    specs = {}

    # img_in / txt_in input projections stay replicated so block activations enter the
    # attention/MLP column-parallel layers replicated (full hidden dim). This keeps the
    # only CCLs the all-reduces after the row-parallel to_out / down projections — no
    # all-gathers, hence no oversized ttnn.concat.

    for block in transformer.transformer_blocks:
        # AdaLN modulation (Sequential(SiLU, Linear)) produces a flat vector that is
        # immediately chunked and applied elementwise to the full hidden dim. Column-
        # sharding its OUTPUT would force an all-gather (lowered to a large ttnn.concat
        # that exceeds Blackhole L1 at opt level 0). Instead ROW-shard it: shard the
        # input dim so each chip computes a partial output that is all-reduced (a
        # supported CCL, no concat) back to a replicated full vector ready for chunking.
        # Replicating it instead OOMs (the modulation is ~6.8B params = ~13.6 GB/chip).
        for mod_name in ("img_mod", "txt_mod"):
            mod = getattr(block, mod_name, None)
            if mod is not None:
                for sub in mod:
                    if isinstance(sub, torch.nn.Linear):
                        _add(specs, sub.weight, (None, "model"))

        _shard_qwenimage_attention(block.attn, specs)
        _shard_qwenimage_feed_forward(block.img_mlp, specs)
        _shard_qwenimage_feed_forward(block.txt_mlp, specs)

    # Final AdaLN: row-shard norm_out.linear (same all-reduce, no concat). proj_out is
    # the small output head and stays replicated.
    if hasattr(transformer.norm_out, "linear"):
        _add(specs, transformer.norm_out.linear.weight, (None, "model"))

    return specs
