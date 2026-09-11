# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Pyramid Flow model loading."""

from typing import Any, Dict, List

import torch

from .flux_modules import PyramidFluxTransformer


# HuggingFace repo carrying the real Pyramid Flow miniFLUX weights (ungated).
_HF_REPO = "rain1011/pyramid-flow-miniflux"
# Variant -> the DiT subfolder inside the repo (each has config.json +
# diffusion_pytorch_model.safetensors). 768p is the default bringup target.
_DIT_SUBFOLDER = "diffusion_transformer_768p"
# The CLIP half of the two-encoder text stack (`text_encoder` / `tokenizer`).
# miniFLUX pairs it with a T5-XXL encoder in `text_encoder_2`, which is not
# exposed here - see `load_text_encoder` for why.
_CLIP_SUBFOLDER = "text_encoder"
_CLIP_TOKENIZER_SUBFOLDER = "tokenizer"
# The CausalVideoVAE that decodes miniFLUX latents back to pixels.
_VAE_SUBFOLDER = "causal_video_vae"


# ============================================================================
# Architectural constants matching `rain1011/pyramid-flow-miniflux`
# (Pyramid Flow miniFLUX-768p DiT, see
#  https://huggingface.co/rain1011/pyramid-flow-miniflux/blob/main/diffusion_transformer_768p/config.json)
# ============================================================================

DIT_CONFIG = dict(
    patch_size=1,
    in_channels=64,
    num_layers=8,
    num_single_layers=16,
    attention_head_dim=64,
    num_attention_heads=30,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    axes_dims_rope=[16, 24, 24],
    use_flash_attn=False,
    use_temporal_causal=True,
    interp_condition_pos=True,
    use_gradient_checkpointing=False,
)


# Internal-patch dimension is hard-coded to 2 in PyramidFluxTransformer; latent
# channels visible to the user are `in_channels // (patch * patch)`.
_INTERNAL_PATCH = 2

# CLIP-L text encoder config, matching
# https://huggingface.co/rain1011/pyramid-flow-miniflux/blob/main/text_encoder/config.json
# (openai/clip-vit-large-patch14). Used only for the offline fallback below.
CLIP_CONFIG = dict(
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=77,
    vocab_size=49408,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-05,
    projection_dim=768,
    bos_token_id=0,
    eos_token_id=2,
    pad_token_id=1,
)

# CausalVideoVAE config deltas against `CausalVideoVAE.__init__` defaults, from
# https://huggingface.co/rain1011/pyramid-flow-miniflux/blob/main/causal_video_vae/config.json
# Every other key in that config.json already matches the constructor default,
# so only these three are needed for the offline fallback below.
VAE_CONFIG = dict(
    encoder_out_channels=16,
    decoder_in_channels=16,
    scaling_factor=0.13025,
)

# Latent geometry: 16 channels, 8x spatial downsample (three spatial upsample
# stages in the decoder), one latent frame.
VAE_LATENT_CHANNELS = 16
VAE_SPATIAL_SCALE = 8
VAE_DEFAULT_RESOLUTION = 256
VAE_LATENT_FRAMES = 1


# The prompt the e2e Pyramid Flow bringup runs use, so device numbers here are
# comparable with the full text-to-video pair.
PROMPT = (
    "A red double-decker bus driving along a sunny coastal road, "
    "waves breaking on the beach below"
)

# Smoke-test latent shape (single pyramid stage).
_SMOKE_TEMP = 1
_SMOKE_HEIGHT = 16
_SMOKE_WIDTH = 16
_SMOKE_TEXT_SEQ_LEN = 16
_SMOKE_BATCH = 1


# ============================================================================
# Model loading
# ============================================================================


def load_transformer(dtype: torch.dtype) -> PyramidFluxTransformer:
    """
    Load the real Pyramid Flow miniFLUX DiT with pretrained weights.

    The weights come from the public (ungated) `rain1011/pyramid-flow-miniflux`
    HuggingFace repo via `from_pretrained`. If the download fails (e.g. no
    network), we fall back to a randomly-initialised model with the corrected
    miniFLUX config so compilation / op-coverage analysis can still run.
    """
    try:
        model = PyramidFluxTransformer.from_pretrained(
            _HF_REPO,
            subfolder=_DIT_SUBFOLDER,
            torch_dtype=dtype,
        )
    except Exception:
        model = PyramidFluxTransformer(**DIT_CONFIG).to(dtype=dtype)
    return model.eval()


class ClipTextEncoderWrapper(torch.nn.Module):
    """Tensors-only CLIP text encoder returning the pooled embedding.

    Mirrors `FluxTextEncoderWithMask._get_clip_prompt_embeds` in the upstream
    Pyramid Flow pipeline: the DiT consumes CLIP's `pooler_output` as its
    `pooled_projections` input, so the pooled vector - not the sequence of
    hidden states - is the tensor worth comparing against CPU.
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, output_hidden_states=False).pooler_output


def load_text_encoder(dtype: torch.dtype) -> ClipTextEncoderWrapper:
    """
    Load the CLIP half of the miniFLUX text stack, wrapped for tensor I/O.

    Only CLIP is exposed. miniFLUX's other encoder is a 4.76B T5-XXL
    (`text_encoder_2`), which compiles and executes on device but lands at
    PCC 0.86 against the CPU reference, so it is not a passing component yet -
    see the loader docstring.

    Falls back to a randomly-initialised model with the real config if the
    weights cannot be downloaded, matching `load_transformer`.
    """
    from transformers import CLIPTextConfig, CLIPTextModel

    try:
        model = CLIPTextModel.from_pretrained(
            _HF_REPO,
            subfolder=_CLIP_SUBFOLDER,
            torch_dtype=dtype,
        )
    except Exception:
        model = CLIPTextModel(CLIPTextConfig(**CLIP_CONFIG)).to(dtype=dtype)
    return ClipTextEncoderWrapper(model.eval()).eval()


class CausalVaeDecoderWrapper(torch.nn.Module):
    """Tensors-only CausalVideoVAE decode returning a plain pixel tensor.

    `pyramid_dit.PyramidDiTForVideoGeneration.decode_latent` decodes with
    `self.vae.decode(latents, temporal_chunk=True, window_size=..., ...)`, so
    `decode` - not the bare `decoder` submodule - is the unit the pipeline
    actually calls and the one worth comparing.

    This wrapper calls the single-shot (`temporal_chunk=False`) path instead.
    The chunked path slices the latent into temporal windows and walks them in
    a Python loop, decoding each with `is_init_image=False` so the causal convs
    carry state across iterations - that loop does not trace. At one latent
    frame it degenerates: `chunk_decode` builds a single-element `frame_list`
    (`full_chunk_size` goes negative, so the loop body never runs) and decodes
    it with `is_init_image=True`, which is what this wrapper does directly. The
    two are bit-exact at `T=1` (verified, max abs diff 0.0), so the traceable
    path is the same math the pipeline runs for a single frame, not an
    approximation of it.

    `return_dict=False` keeps graph capture on a plain tensor rather than a
    `DecoderOutput` dataclass.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(
            latents, is_init_image=True, temporal_chunk=False, return_dict=False
        )[0]


def load_vae_decoder(dtype: torch.dtype) -> CausalVaeDecoderWrapper:
    """
    Load the CausalVideoVAE decode half, wrapped for tensor I/O.

    The encoder half is dropped after loading: `decode` never touches it, and
    keeping it would double the resident parameters of a component whose only
    job is latents -> pixels.

    Falls back to a randomly-initialised model with the real config if the
    weights cannot be downloaded, matching `load_transformer`.
    """
    from .video_vae import CausalVideoVAE

    try:
        vae = CausalVideoVAE.from_pretrained(
            _HF_REPO,
            subfolder=_VAE_SUBFOLDER,
            torch_dtype=dtype,
        )
    except Exception:
        vae = CausalVideoVAE(**VAE_CONFIG).to(dtype=dtype)

    # Decode-only component: drop the encoder tower and its quant conv.
    vae.encoder = None
    vae.quant_conv = None
    return CausalVaeDecoderWrapper(vae.eval()).eval()


# ============================================================================
# Input loading
# ============================================================================


def load_transformer_inputs(dtype: torch.dtype) -> Dict[str, Any]:
    """
    Build synthetic inputs for a single-stage Pyramid Flow DiT forward pass.

    Returns a dict matching `PyramidFluxTransformer.forward` signature.
    The `sample` field is a list-of-lists per the upstream pyramid-stage
    structure: `[stage_0_clips, stage_1_clips, ...]` where each clip is a
    `[B, C_latent, T, H, W]` tensor. We use a single stage with a single
    clip for the smoke variant.
    """
    cfg = DIT_CONFIG
    batch_size = _SMOKE_BATCH
    latent_channels = cfg["in_channels"] // (_INTERNAL_PATCH * _INTERNAL_PATCH)
    seq_len = _SMOKE_TEXT_SEQ_LEN

    sample = [
        [
            torch.randn(
                batch_size,
                latent_channels,
                _SMOKE_TEMP,
                _SMOKE_HEIGHT,
                _SMOKE_WIDTH,
                dtype=dtype,
            )
        ]
    ]
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, cfg["joint_attention_dim"], dtype=dtype
    )
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    pooled_projections = torch.randn(
        batch_size, cfg["pooled_projection_dim"], dtype=dtype
    )
    timestep_ratio = torch.tensor([500.0], dtype=dtype)

    return {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "pooled_projections": pooled_projections,
        "timestep_ratio": timestep_ratio,
    }


def load_text_encoder_inputs(dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Tokenize the bringup prompt for the CLIP text encoder.

    `dtype` is unused - CLIP takes integer token ids - but the signature is kept
    uniform with `load_transformer_inputs`. Input ids are padded to CLIP's fixed
    77-token context, which is what the upstream pipeline does.

    If the tokenizer cannot be downloaded, falls back to deterministic synthetic
    ids so compilation / op-coverage analysis still runs offline.
    """
    max_length = CLIP_CONFIG["max_position_embeddings"]
    try:
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            _HF_REPO, subfolder=_CLIP_TOKENIZER_SUBFOLDER
        )
        input_ids = tokenizer(
            [PROMPT],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
    except Exception:
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(
            0,
            CLIP_CONFIG["vocab_size"],
            (_SMOKE_BATCH, max_length),
            generator=generator,
        )
    return [input_ids]


def load_vae_decoder_inputs(dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Build a single-frame latent for the CausalVideoVAE decoder.

    Shape is `[B, 16, T, H, W]` - the 5-D latent the DiT emits and the VAE
    consumes - at one temporal frame and `VAE_DEFAULT_RESOLUTION / 8` spatially,
    so the decode lands on a 256x256 frame.

    The generator is seeded so two processes comparing this component see the
    same latent; without that, a CPU-vs-device comparison silently measures two
    different inputs. The draw is always made in fp32 and cast afterwards, for
    the same reason: `torch.randn` consumes the generator differently per dtype,
    so drawing directly at `dtype=torch.bfloat16` yields a tensor unrelated to
    the fp32 draw from the same seed (measured PCC 0.0098). Casting keeps the
    latent dtype-independent, so an fp32 CPU reference and a bf16 device run
    decode the same values.
    """
    latent_hw = VAE_DEFAULT_RESOLUTION // VAE_SPATIAL_SCALE
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(
        _SMOKE_BATCH,
        VAE_LATENT_CHANNELS,
        VAE_LATENT_FRAMES,
        latent_hw,
        latent_hw,
        dtype=torch.float32,
        generator=generator,
    )
    return [latents.to(dtype)]
