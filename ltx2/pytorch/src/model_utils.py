# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Lightricks/LTX-2 (LTX2Pipeline).

LTX-2 is an audiovisual text/image-to-video diffusion pipeline. It is brought
up by composite components rather than as a single graph:

  - TextEncoder  -> Gemma3ForConditionalGeneration  (~12B, gemma3_text)
  - Connectors   -> LTX2TextConnectors               (~1.4B)
  - Transformer  -> LTX2VideoTransformer3DModel      (~19B, audiovisual DiT denoiser)
  - Vae          -> AutoencoderKLLTX2Video decoder   (~1.2B)

Bringup resolution: 128x128 pixels, 9 frames @ 24fps (matches the small
component-test resolution convention used for the very large diffusion
pipelines, e.g. flux2). The composite end-to-end generation targets the
pipeline's native default resolution; see the report.
"""

import torch

REPO_ID = "Lightricks/LTX-2"
DTYPE = torch.bfloat16

PROMPT = (
    "A cinematic shot of a fluffy corgi running across a sunlit meadow, "
    "slow motion, shallow depth of field."
)

# ---------------------------------------------------------------------------
# Bringup geometry (small resolution for per-component compile/run validation)
# ---------------------------------------------------------------------------
HEIGHT = 128
WIDTH = 128
NUM_FRAMES = 9
FRAME_RATE = 24.0
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 4.0
SEED = 42
MAX_SEQUENCE_LENGTH = 128

# VAE compression (vae/config.json)
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8

# Transformer config (transformer/config.json)
CAPTION_CHANNELS = 3840
TRANSFORMER_IN_CHANNELS = 128
AUDIO_IN_CHANNELS = 128
PATCH_SIZE = 1
PATCH_SIZE_T = 1

# Text encoder (text_encoder/config.json -> text_config: gemma3_text)
TEXT_HIDDEN_SIZE = 3840
TEXT_NUM_LAYERS = 48
# output_hidden_states returns num_layers + 1 (embeddings + each layer output)
TEXT_NUM_HIDDEN_STATES = TEXT_NUM_LAYERS + 1
# Packed prompt-embed dim consumed by the connectors:
PROMPT_EMBED_DIM = TEXT_HIDDEN_SIZE * TEXT_NUM_HIDDEN_STATES  # 3840 * 49

# Audio latent geometry (audio_vae/config.json + transformer audio config)
AUDIO_SAMPLING_RATE = 16000
AUDIO_HOP_LENGTH = 160
AUDIO_TEMPORAL_COMPRESSION = 4
AUDIO_MEL_BINS = 64
AUDIO_MEL_COMPRESSION = 4
AUDIO_LATENT_CHANNELS = 8


def latent_video_geometry(
    height: int = HEIGHT, width: int = WIDTH, num_frames: int = NUM_FRAMES
):
    """Return (latent_frames, latent_height, latent_width) for the video VAE."""
    lf = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION + 1
    lh = height // VAE_SPATIAL_COMPRESSION
    lw = width // VAE_SPATIAL_COMPRESSION
    return lf, lh, lw


def audio_num_frames(num_frames: int = NUM_FRAMES, frame_rate: float = FRAME_RATE):
    """Number of audio latent frames matching the video duration."""
    duration_s = num_frames / frame_rate
    audio_latents_per_second = (
        AUDIO_SAMPLING_RATE / AUDIO_HOP_LENGTH / float(AUDIO_TEMPORAL_COMPRESSION)
    )
    return round(duration_s * audio_latents_per_second)


# (batch, model) mesh shapes by device count (same convention as flux2)
MESH_SHAPES = {32: (8, 4), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------
def load_text_encoder(dtype: torch.dtype = DTYPE):
    from transformers import Gemma3ForConditionalGeneration

    return Gemma3ForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer")


def load_connectors(dtype: torch.dtype = DTYPE):
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    return LTX2TextConnectors.from_pretrained(
        REPO_ID,
        subfolder="connectors",
        torch_dtype=dtype,
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    from diffusers import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    from diffusers import AutoencoderKLLTX2Video

    return AutoencoderKLLTX2Video.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


def tokenize_prompt(
    prompt: str = PROMPT, *, max_sequence_length: int = MAX_SEQUENCE_LENGTH
):
    """Tokenize a prompt the way LTX2Pipeline._get_gemma_prompt_embeds does."""
    tokenizer = load_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [prompt.strip()],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return inputs.input_ids, inputs.attention_mask


# ---------------------------------------------------------------------------
# Wrappers (tensor-in / tensor-out, mirroring the LTX2Pipeline call sites)
# ---------------------------------------------------------------------------
class Gemma3TextEncoderWrapper(torch.nn.Module):
    """Mirror LTX2Pipeline._get_gemma_prompt_embeds: stack all hidden states."""

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
            use_cache=False,
        )
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)  # [B, S, H, L+1]
        return hidden_states.flatten(2, 3)  # [B, S, H*(L+1)]


class LTX2ConnectorsWrapper(torch.nn.Module):
    """Project packed Gemma3 prompt embeds into the transformer caption space."""

    def __init__(self, connectors, padding_side: str = "left"):
        super().__init__()
        self.connectors = connectors
        self.padding_side = padding_side

    def forward(
        self, prompt_embeds: torch.Tensor, prompt_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        connector_prompt_embeds, _, _ = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=self.padding_side
        )
        return connector_prompt_embeds


class LTX2TransformerWrapper(torch.nn.Module):
    """Single audiovisual denoise step of LTX2VideoTransformer3DModel.

    RoPE coordinates depend only on the (fixed) bringup geometry, so they are
    precomputed once at construction and stored as buffers rather than traced.
    """

    def __init__(
        self,
        transformer,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_frames: int = NUM_FRAMES,
        frame_rate: float = FRAME_RATE,
    ):
        super().__init__()
        self.transformer = transformer
        self.frame_rate = frame_rate
        lf, lh, lw = latent_video_geometry(height, width, num_frames)
        self.latent_frames = lf
        self.latent_height = lh
        self.latent_width = lw
        self.audio_frames = audio_num_frames(num_frames, frame_rate)

        with torch.no_grad():
            video_coords = transformer.rope.prepare_video_coords(
                1, lf, lh, lw, torch.device("cpu"), fps=frame_rate
            )
            audio_coords = transformer.audio_rope.prepare_audio_coords(
                1, self.audio_frames, torch.device("cpu")
            )
        self.register_buffer("video_coords", video_coords, persistent=False)
        self.register_buffer("audio_coords", audio_coords, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        noise_pred_video, _ = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            sigma=timestep,
            num_frames=self.latent_frames,
            height=self.latent_height,
            width=self.latent_width,
            fps=self.frame_rate,
            audio_num_frames=self.audio_frames,
            video_coords=self.video_coords,
            audio_coords=self.audio_coords,
            return_dict=False,
        )
        return noise_pred_video


class LTX2VaeDecoderWrapper(torch.nn.Module):
    """Decode video latents [B, C, F, H, W] -> pixels with AutoencoderKLLTX2Video."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Synthetic inputs for each component at bringup resolution
# ---------------------------------------------------------------------------
def make_video_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Packed video latents [B, num_video_tokens, in_channels]."""
    gen = torch.Generator().manual_seed(SEED)
    lf, lh, lw = latent_video_geometry()
    num_tokens = lf * lh * lw
    return torch.randn(
        1, num_tokens, TRANSFORMER_IN_CHANNELS, dtype=dtype, generator=gen
    )


def make_audio_latents(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Packed audio latents [B, num_audio_tokens, audio_in_channels]."""
    gen = torch.Generator().manual_seed(SEED + 1)
    af = audio_num_frames()
    return torch.randn(1, af, AUDIO_IN_CHANNELS, dtype=dtype, generator=gen)


def make_caption_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Connector-projected text conditioning [B, text_seq, caption_channels]."""
    gen = torch.Generator().manual_seed(SEED + 2)
    return torch.randn(
        1, MAX_SEQUENCE_LENGTH, CAPTION_CHANNELS, dtype=dtype, generator=gen
    )


def make_packed_prompt_embeds(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Stacked Gemma3 hidden states [B, text_seq, H*(L+1)] (connector input)."""
    gen = torch.Generator().manual_seed(SEED + 3)
    return torch.randn(
        1, MAX_SEQUENCE_LENGTH, PROMPT_EMBED_DIM, dtype=dtype, generator=gen
    )


def make_prompt_attention_mask() -> torch.Tensor:
    return torch.ones(1, MAX_SEQUENCE_LENGTH, dtype=torch.int64)


def make_vae_decoder_input(dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """Unpacked video latent grid [B, C, F, H, W] for the VAE decoder."""
    gen = torch.Generator().manual_seed(SEED + 4)
    lf, lh, lw = latent_video_geometry()
    return torch.randn(
        1, TRANSFORMER_IN_CHANNELS, lf, lh, lw, dtype=dtype, generator=gen
    )


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------
def _add_shard_spec(specs: dict, param, spec: tuple) -> None:
    if param is not None:
        specs[param] = spec


def shard_text_encoder_specs(model) -> dict:
    """Column/row-parallel shard spec for the Gemma3 language model."""
    specs = {}
    language_model = model
    for attr in ("model", "language_model"):
        if hasattr(language_model, attr):
            language_model = getattr(language_model, attr)
    # Gemma3ForConditionalGeneration -> .model.language_model.layers
    if hasattr(language_model, "language_model"):
        language_model = language_model.language_model

    if hasattr(language_model, "embed_tokens"):
        _add_shard_spec(specs, language_model.embed_tokens.weight, (None, None))

    layers = getattr(language_model, "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name, None)
            if proj is not None:
                _add_shard_spec(specs, proj.weight, ("model", None))
                _add_shard_spec(specs, getattr(proj, "bias", None), ("model",))
        if getattr(sa, "o_proj", None) is not None:
            _add_shard_spec(specs, sa.o_proj.weight, (None, "model"))
            _add_shard_spec(specs, getattr(sa.o_proj, "bias", None), (None,))
        for norm_name in ("q_norm", "k_norm"):
            norm = getattr(sa, norm_name, None)
            if norm is not None:
                _add_shard_spec(specs, getattr(norm, "weight", None), (None,))

        mlp = layer.mlp
        _add_shard_spec(specs, mlp.gate_proj.weight, ("model", None))
        _add_shard_spec(specs, mlp.up_proj.weight, ("model", None))
        _add_shard_spec(specs, mlp.down_proj.weight, (None, "model"))

        for norm_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            norm = getattr(layer, norm_name, None)
            if norm is not None:
                _add_shard_spec(specs, getattr(norm, "weight", None), (None,))

    if hasattr(language_model, "norm"):
        _add_shard_spec(specs, language_model.norm.weight, (None,))

    return specs


def _shard_ltx2_attention(attn, specs: dict) -> None:
    """Megatron column/row TP for a single LTX2Attention module.

    to_q/to_k/to_v are column-parallel (split on head dim); to_out[0] is
    row-parallel. The `rms_norm_across_heads` qk-norms operate over the full
    (sharded) inner dim, so their affine weights are sharded to match — GSPMD
    inserts the cross-head all-reduce for the RMS statistic.
    """
    if attn is None:
        return
    for n in ("to_q", "to_k", "to_v"):
        proj = getattr(attn, n, None)
        if isinstance(proj, torch.nn.Linear):
            _add_shard_spec(specs, proj.weight, ("model", None))
            _add_shard_spec(specs, getattr(proj, "bias", None), ("model",))
    for norm_name in ("norm_q", "norm_k"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            _add_shard_spec(specs, getattr(norm, "weight", None), ("model",))
    to_out = getattr(attn, "to_out", None)
    if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0:
        _add_shard_spec(specs, to_out[0].weight, (None, "model"))
        _add_shard_spec(specs, getattr(to_out[0], "bias", None), (None,))


def _shard_ltx2_feed_forward(ff, specs: dict) -> None:
    """Column-parallel input proj, row-parallel output proj for a FeedForward."""
    net = getattr(ff, "net", None)
    if net is None:
        return
    for sub in net:
        proj = getattr(sub, "proj", None)  # GELU/GEGLU activation wraps a Linear
        if isinstance(proj, torch.nn.Linear):
            _add_shard_spec(specs, proj.weight, ("model", None))
            _add_shard_spec(specs, getattr(proj, "bias", None), ("model",))
        elif isinstance(sub, torch.nn.Linear):
            _add_shard_spec(specs, sub.weight, (None, "model"))
            _add_shard_spec(specs, getattr(sub, "bias", None), (None,))


def shard_transformer_specs(transformer) -> dict:
    """Tensor-parallel shard spec for LTX2VideoTransformer3DModel.

    Each LTX2VideoTransformerBlock has six attention modules (video/audio
    self-attn, video/audio cross-attn, and the two cross-modal attns) plus a
    video and audio FeedForward. All are sharded on the "model" axis; norms,
    modulation, caption projections and the in/out projections stay replicated.
    """
    specs = {}

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return specs

    for block in blocks:
        for attn_name in (
            "attn1",
            "audio_attn1",
            "attn2",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ):
            _shard_ltx2_attention(getattr(block, attn_name, None), specs)
        for ff_name in ("ff", "audio_ff"):
            ff = getattr(block, ff_name, None)
            if ff is not None:
                _shard_ltx2_feed_forward(ff, specs)

    return specs
