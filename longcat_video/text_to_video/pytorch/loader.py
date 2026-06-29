#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video text-to-video model loader implementation.

LongCat-Video (meituan-longcat/LongCat-Video) is a 13.6B-parameter diffusion
video-generation pipeline. Like other diffusion pipelines it is brought up by
independently-compilable *components* rather than as a single graph:

- ``text_encoder`` : UMT5-xxl encoder (transformers ``UMT5EncoderModel``)
- ``vae``          : Wan 3D VAE (diffusers ``AutoencoderKLWan``)
- ``dit``          : ``LongCatVideoTransformer3DModel`` — the 48-layer DiT
                     denoiser, the key component (vendored under ``src/`` from
                     the LongCat-Video GitHub repo, with the CUDA-only
                     flash-attn / xformers / block-sparse / context-parallel
                     paths replaced by portable SDPA).

Select a component with the ``subfolder`` argument; ``dit`` is the default
(key) component.

Native text-to-video resolution (from ``run_demo_text_to_video.py``): 480x832,
93 frames, 50 steps, all components in bfloat16.
"""

from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# Repo + native generation parameters (see run_demo_text_to_video.py).
_REPO = "meituan-longcat/LongCat-Video"
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 832
NATIVE_NUM_FRAMES = 93
NATIVE_STEPS = 50
MAX_TEXT_TOKENS = 512
CAPTION_CHANNELS = 4096  # text_encoder d_model == dit caption_channels
LATENT_CHANNELS = 16
VAE_SPATIAL_DOWNSAMPLE = 8  # AutoencoderKLWan: 2 ** len(temperal_downsample-ish dim_mult)
VAE_TEMPORAL_DOWNSAMPLE = 4

SUPPORTED_SUBFOLDERS = {"text_encoder", "vae", "dit"}

DEFAULT_PROMPT = (
    "In a realistic photography style, a white boy around seven or eight years "
    "old sits on a park bench, wearing a light blue T-shirt, denim shorts, and "
    "white sneakers. He holds an ice cream cone with vanilla and chocolate "
    "flavors, and beside him is a medium-sized golden Labrador."
)


def _latent_dims(height, width, num_frames):
    """Native latent dims [T, H, W] fed to the DiT / produced by the VAE."""
    t = (num_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
    h = height // VAE_SPATIAL_DOWNSAMPLE
    w = width // VAE_SPATIAL_DOWNSAMPLE
    return t, h, w


class ModelVariant(StrEnum):
    """Available LongCat-Video variants."""

    BASE = "13.6b"


class ModelLoader(ForgeModel):
    """LongCat-Video component loader (text_encoder / vae / dit)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(pretrained_model_name=_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: str = "dit",
    ):
        super().__init__(variant)
        if subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LongCat-Video",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #
    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        """Load the selected component. Default dtype is bfloat16 (native)."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        name = self._variant_config.pretrained_model_name

        if self._subfolder == "text_encoder":
            from transformers import UMT5EncoderModel

            model = UMT5EncoderModel.from_pretrained(
                name, subfolder="text_encoder", torch_dtype=dtype
            )
            return model.eval()

        if self._subfolder == "vae":
            from diffusers import AutoencoderKLWan

            model = AutoencoderKLWan.from_pretrained(
                name, subfolder="vae", torch_dtype=dtype
            )
            return model.eval()

        # dit (key component) — vendored custom transformer
        from .src.longcat_video_dit import LongCatVideoTransformer3DModel

        model = LongCatVideoTransformer3DModel.from_pretrained(
            name,
            subfolder="dit",
            torch_dtype=dtype,
            # force the portable SDPA path; disable CUDA-only kernels
            cp_split_hw=[1, 1],
            enable_flashattn2=False,
            enable_flashattn3=False,
            enable_xformers=False,
            enable_bsa=False,
        )

        # The model computes timestep/AdaLN modulation in fp32, relying on a
        # `torch.amp.autocast(device_type="cuda", dtype=float32)` context to
        # upcast the relevant Linear matmuls. That autocast is a no-op off CUDA
        # (CPU / XLA), which leaves fp32 activations feeding bf16 weights and
        # raises a dtype-mismatch. Keeping these small precision-sensitive
        # submodules in fp32 reproduces the intended GPU behavior in a
        # device-agnostic way (the AdaLN/timestep Linears are <5% of params).
        if dtype != torch.float32:
            model.t_embedder.float()
            model.final_layer.adaLN_modulation.float()
            for block in model.blocks:
                block.adaLN_modulation.float()

        return model.eval()

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, subfolder="tokenizer"
            )
        return self._tokenizer

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        *,
        prompt: Optional[str] = None,
        height: int = NATIVE_HEIGHT,
        width: int = NATIVE_WIDTH,
        num_frames: int = NATIVE_NUM_FRAMES,
        batch_size: int = 2,  # CFG: [uncond, cond]
        vae_type: str = "decoder",
        seq_len: int = MAX_TEXT_TOKENS,
    ):
        """Build sample inputs for the selected component at native resolution.

        height/width/num_frames may be reduced for a cheap CPU loader-sanity
        forward; the device gate uses the native defaults.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "text_encoder":
            tok = self._get_tokenizer()
            enc = tok(
                prompt or DEFAULT_PROMPT,
                padding="max_length",
                max_length=seq_len,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc.input_ids,
                "attention_mask": enc.attention_mask,
            }

        t_lat, h_lat, w_lat = _latent_dims(height, width, num_frames)

        if self._subfolder == "vae":
            if vae_type == "encoder":
                # raw video frames [B, C=3, T, H, W] in [-1, 1]
                video = torch.randn(1, 3, num_frames, height, width, dtype=dtype)
                return {"x": video}
            # decoder: latents [B, 16, T_lat, H_lat, W_lat]
            latents = torch.randn(1, LATENT_CHANNELS, t_lat, h_lat, w_lat, dtype=dtype)
            return {"z": latents}

        # dit
        hidden_states = torch.randn(
            batch_size, LATENT_CHANNELS, t_lat, h_lat, w_lat, dtype=dtype
        )
        timestep = torch.full((batch_size,), 1000.0, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, 1, seq_len, CAPTION_CHANNELS, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
