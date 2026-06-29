#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video DiT denoiser component loader (the key component).

LongCat-Video's denoiser is a custom 13.6B 3D diffusion transformer
(`LongCatVideoTransformer3DModel`, depth=48, hidden=4096, 32 heads, patch
[1,2,2], 16 latent channels) with 3D RoPE self-attention and varlen text
cross-attention. The modeling code lives in the model's GitHub repo
(github.com/meituan-longcat/LongCat-Video), not on the Hub, so it is vendored
under `src/modeling/` with three bringup-oriented changes:

  * flash-attn-2/3, xformers and block-sparse attention are replaced by a
    device-friendly `torch.nn.functional.scaled_dot_product_attention` fallback
    (all `enable_*` flags forced False);
  * the context-parallel (ulysses) split/gather is dropped (single device,
    cp_split_hw == [1, 1]);
  * the CUDA-only `amp.autocast` fp32 regions become no-ops so the graph is
    device-agnostic.

`load_inputs` uses encoder_attention_mask=None so the text tokens are not packed
via a data-dependent `masked_select` (which would create dynamic shapes); the
full 512 conditioning tokens are kept, matching `text_tokens_zero_pad`.

Inputs are sized at the native t2v resolution (480x832, 93 frames -> latent
[16, 24, 60, 104], sequence length 37440). Batch is 1 here; the real pipeline
runs batch 2 for classifier-free guidance.

Available variants:
- LONGCAT_VIDEO: meituan-longcat/LongCat-Video (dit subfolder)
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
from .src.modeling.longcat_video_dit import LongCatVideoTransformer3DModel

# Native t2v generation resolution (run_demo_text_to_video.py defaults).
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 832
NATIVE_FRAMES = 93
SPATIAL_DOWNSAMPLE = 8
TEMPORAL_DOWNSAMPLE = 4

IN_CHANNELS = 16
CAPTION_CHANNELS = 4096
MAX_SEQUENCE_LENGTH = 512


def _native_latent_shape():
    t = (NATIVE_FRAMES - 1) // TEMPORAL_DOWNSAMPLE + 1  # 24
    h = NATIVE_HEIGHT // SPATIAL_DOWNSAMPLE  # 60
    w = NATIVE_WIDTH // SPATIAL_DOWNSAMPLE  # 104
    return t, h, w


class ModelVariant(StrEnum):
    """Available LongCat-Video denoiser variants."""

    LONGCAT_VIDEO = "longcat_video"


class ModelLoader(ForgeModel):
    """LongCat-Video DiT denoiser loader."""

    _VARIANTS = {
        ModelVariant.LONGCAT_VIDEO: ModelConfig(
            pretrained_model_name="meituan-longcat/LongCat-Video",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LONGCAT_VIDEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="longcat_video_dit",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        """Load the DiT denoiser with the SDPA attention fallback enabled."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = LongCatVideoTransformer3DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="dit",
            torch_dtype=dtype,
            # Force the device-friendly SDPA path; single device (no CP).
            enable_flashattn2=False,
            enable_flashattn3=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=[1, 1],
        )
        model.eval()
        return model

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        latent_t: Optional[int] = None,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
    ):
        """DiT inputs at native latent resolution (encoder_attention_mask=None)."""
        nt, nh, nw = _native_latent_shape()
        t = latent_t if latent_t is not None else nt
        h = latent_h if latent_h is not None else nh
        w = latent_w if latent_w is not None else nw
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        hidden_states = torch.randn(batch_size, IN_CHANNELS, t, h, w, dtype=dtype)
        timestep = torch.full((batch_size,), 1000.0, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, 1, MAX_SEQUENCE_LENGTH, CAPTION_CHANNELS, dtype=dtype
        )
        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            # encoder_attention_mask omitted (=None): keep all 512 tokens, avoid
            # the data-dependent masked_select / varlen packing path.
        }
