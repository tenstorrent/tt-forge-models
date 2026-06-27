# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Component loaders / input builders for the LTX-2.3 audio-video diffusion pipeline.

LTX-2.3 (`Lightricks/LTX-2.3`) is a ~22B-parameter DiT-based *joint audio-video*
foundation model.  The HuggingFace repo ships the model as a single bundled
native checkpoint (``ltx-2.3-22b-*.safetensors``) that contains the transformer
weights *plus* the (audio) VAE and vocoder under a ``model.diffusion_model.``
prefix; there is no diffusers-format folder layout and no standalone VAE /
text-encoder folder in the 2.3 repo.

diffusers (>=0.38) recognises this checkpoint as ``ltx2`` and, via
``LTX2VideoTransformer3DModel.from_single_file``, strips the prefix and pulls the
*architecture config* from ``Lightricks/LTX-2`` (the previous, diffusers-format
release).  The 2.3 transformer is architecturally identical to LTX-2
(``LTX2VideoTransformer3DModel``, 48 layers, hidden 4096, caption dim 3840) - only
the weights differ.  The VAE (``AutoencoderKLLTX2Video``) and the text encoder
(``Gemma3ForConditionalGeneration``) are reused from ``Lightricks/LTX-2``.

This module decomposes the pipeline into the three independently-compilable
components a tt-forge bringup validates:

  * ``transformer``    - the per-denoise-step DiT (the heavy sharding target)
  * ``vae``            - ``AutoencoderKLLTX2Video`` (latent <-> pixel video)
  * ``text_encoder``   - ``Gemma3ForConditionalGeneration`` prompt encoder
"""

import torch

# --------------------------------------------------------------------------- #
# Repos / checkpoints
# --------------------------------------------------------------------------- #
# The target model (2.3 weights, bundled single-file checkpoint).
LTX23_REPO = "Lightricks/LTX-2.3"
# Distilled variant: 8 steps, CFG=1 - the cheapest faithful generation path.
DISTILLED_CKPT = "ltx-2.3-22b-distilled.safetensors"
DEV_CKPT = "ltx-2.3-22b-dev.safetensors"
# Architecture-identical diffusers-format release. Source of the transformer
# *config*, and of the VAE + text-encoder weights (2.3 ships neither standalone).
LTX2_REPO = "Lightricks/LTX-2"

DTYPE = torch.bfloat16

# --------------------------------------------------------------------------- #
# Tensor-parallel mesh (Megatron column->row). mesh = (batch=1, model=num_dev).
# Video & audio attention both have 32 heads, divisible by 1/2/4/8.
# --------------------------------------------------------------------------- #
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
MESH_NAMES = (None, "model")

# --------------------------------------------------------------------------- #
# Native generation geometry (i2v pipeline __call__ defaults)
#   height=512, width=768, num_frames=121, fps=24
# VAE: spatial_compression=32, temporal_compression=8, patch_size=4
# --------------------------------------------------------------------------- #
HEIGHT = 512
WIDTH = 768
NUM_FRAMES = 121
FPS = 24.0

SPATIAL_COMPRESSION = 32
TEMPORAL_COMPRESSION = 8
LATENT_CHANNELS = 128  # vae latent_channels / transformer in_channels

LATENT_H = HEIGHT // SPATIAL_COMPRESSION  # 16
LATENT_W = WIDTH // SPATIAL_COMPRESSION  # 24
LATENT_T = (NUM_FRAMES - 1) // TEMPORAL_COMPRESSION + 1  # 16

# Transformer text-conditioning (Gemma-3 hidden size).
CAPTION_CHANNELS = 3840
TEXT_SEQ_LEN = 128

# Audio modality. ~24kHz mel latents downsampled in time; ~25 latent frames/s.
# duration = NUM_FRAMES / FPS ~= 5.04s -> ~126 audio latent frames.
AUDIO_IN_CHANNELS = 128
AUDIO_NUM_FRAMES = round(NUM_FRAMES / FPS * 25)  # 126


# --------------------------------------------------------------------------- #
# Component loaders
# --------------------------------------------------------------------------- #
def load_transformer(dtype=DTYPE, use_reference_weights=False):
    """Load the LTX-2.3 DiT denoiser as an ``LTX2VideoTransformer3DModel``.

    Args:
        dtype: weight dtype.
        use_reference_weights: when ``True``, load the architecture-identical
            LTX-2 transformer (diffusers format) from ``Lightricks/LTX-2`` instead
            of the 22B 2.3 single-file checkpoint. The 2.3 ``from_single_file``
            path requires the full ~46GB bundled checkpoint to be present; the
            reference path lets the *architecture* be exercised when that
            download is unavailable. The two are the same diffusers class and
            config; only the trained weights differ.
    """
    from diffusers import LTX2VideoTransformer3DModel

    if use_reference_weights:
        model = LTX2VideoTransformer3DModel.from_pretrained(
            LTX2_REPO, subfolder="transformer", torch_dtype=dtype
        )
    else:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(LTX23_REPO, DISTILLED_CKPT)
        # diffusers detects the `ltx2` checkpoint and fetches the matching
        # architecture config from `Lightricks/LTX-2`.
        model = LTX2VideoTransformer3DModel.from_single_file(
            ckpt, config=LTX2_REPO, subfolder="transformer", torch_dtype=dtype
        )
    return model.eval()


def load_vae(dtype=DTYPE):
    """Load ``AutoencoderKLLTX2Video`` (reused from the LTX-2 release)."""
    from diffusers import AutoencoderKLLTX2Video

    vae = AutoencoderKLLTX2Video.from_pretrained(
        LTX2_REPO, subfolder="vae", torch_dtype=dtype
    )
    return vae.eval()


def load_text_encoder(dtype=DTYPE):
    """Load the Gemma-3 prompt encoder (``Gemma3ForConditionalGeneration``)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        LTX2_REPO, subfolder="text_encoder", torch_dtype=dtype
    )
    return model.eval()


# --------------------------------------------------------------------------- #
# Wrappers - tensor-in / tensor-out for the compiler
# --------------------------------------------------------------------------- #
class LTX2TransformerWrapper(torch.nn.Module):
    """Simplify the joint A/V transformer forward to positional tensor I/O.

    RoPE ``video_coords`` / ``audio_coords`` are computed inside the transformer
    from ``num_frames`` / ``height`` / ``width`` / ``audio_num_frames`` (passed as
    Python ints), so they are not graph inputs. Returns the denoised video latent
    patch sequence (first element of the tuple output).
    """

    def __init__(self, transformer, num_frames, height, width, audio_num_frames, fps=FPS):
        super().__init__()
        self.transformer = transformer
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.audio_num_frames = audio_num_frames
        self.fps = fps

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
    ):
        out = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=audio_timestep,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            audio_num_frames=self.audio_num_frames,
            fps=self.fps,
            return_dict=False,
        )
        return out[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose ``AutoencoderKLLTX2Video`` decode as ``(z) -> pixel video``."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def load_transformer_inputs(dtype=DTYPE):
    """Positional inputs for ``LTX2TransformerWrapper`` at native i2v geometry.

    Shapes:
      hidden_states               [1, T*H*W, 128]   (patchified video latents)
      audio_hidden_states         [1, A,    128]    (patchified audio latents)
      encoder_hidden_states       [1, S,   3840]    (video prompt embeds)
      audio_encoder_hidden_states [1, S,   3840]    (audio prompt embeds)
      timestep                    [1, T*H*W]        (per-token, scaled)
    """
    num_video_tokens = LATENT_T * LATENT_H * LATENT_W  # 6144
    hidden_states = torch.randn(1, num_video_tokens, LATENT_CHANNELS, dtype=dtype)
    audio_hidden_states = torch.randn(1, AUDIO_NUM_FRAMES, AUDIO_IN_CHANNELS, dtype=dtype)
    encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype)
    audio_encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, CAPTION_CHANNELS, dtype=dtype)
    # timestep scaled by timestep_scale_multiplier (1000); mid-noise step.
    # video timestep is per video token; audio timestep is per audio token.
    timestep = torch.full((1, num_video_tokens), 500_000.0, dtype=dtype)
    audio_timestep = torch.full((1, AUDIO_NUM_FRAMES), 500_000.0, dtype=dtype)
    return [
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
    ]


def load_vae_decoder_inputs(dtype=DTYPE):
    """Latent input for the VAE decoder: [1, 128, T, H, W]."""
    return torch.randn(1, LATENT_CHANNELS, LATENT_T, LATENT_H, LATENT_W, dtype=dtype)


def load_vae_encoder_inputs(dtype=DTYPE):
    """Pixel-video input for the VAE encoder: [1, 3, NUM_FRAMES, HEIGHT, WIDTH]."""
    return torch.randn(1, 3, NUM_FRAMES, HEIGHT, WIDTH, dtype=dtype)


def shard_transformer_specs(transformer) -> dict:
    """Megatron column->row tensor-parallel spec for ``LTX2VideoTransformer3DModel``.

    Column-parallel (Q/K/V, FF up):  weight ("model", None), bias ("model",)
    Row-parallel    (out, FF down):  weight (None, "model"), bias replicated (None,)

    Applied to every per-block attention (video self ``attn1``, audio self
    ``audio_attn1``, video text-cross ``attn2``, audio text-cross ``audio_attn2``,
    and the two A/V cross-attentions) plus the video/audio feed-forwards - this is
    where essentially all ~19-22B parameters live. AdaLN tables, projections and
    norms are left replicated.
    """
    specs = {}

    def col(linear):
        specs[linear.weight] = ("model", None)
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = ("model",)

    def row(linear):
        specs[linear.weight] = (None, "model")
        if getattr(linear, "bias", None) is not None:
            specs[linear.bias] = (None,)

    attn_names = [
        "attn1",
        "audio_attn1",
        "attn2",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    ]
    for block in transformer.transformer_blocks:
        for name in attn_names:
            attn = getattr(block, name, None)
            if attn is None:
                continue
            col(attn.to_q)
            col(attn.to_k)
            col(attn.to_v)
            row(attn.to_out[0])
        # Feed-forwards: net[0].proj is column-parallel, net[2] is row-parallel.
        for ff_name in ("ff", "audio_ff"):
            ff = getattr(block, ff_name, None)
            if ff is None:
                continue
            col(ff.net[0].proj)
            row(ff.net[2])
    return specs


def load_text_encoder_inputs(dtype=DTYPE):
    """Token inputs for the Gemma-3 text encoder: input_ids + attention_mask."""
    input_ids = torch.randint(0, 262_208, (1, TEXT_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
