# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance SeedVR2-7B video restoration model loader implementation.

SeedVR2 is a one-step diffusion-based video restoration model using a NaDiT
(Neighborhood Attention Diffusion Transformer) architecture. It takes degraded
low-quality video frames and produces high-quality restored output.

Available variants:
- SEEDVR2_7B: Standard 7B parameter model (seedvr2_ema_7b.pth)
- SEEDVR2_7B_SHARP: Sharper variant (seedvr2_ema_7b_sharp.pth)
"""

import os
import sys
import types
from typing import Optional

import torch
from huggingface_hub import snapshot_download

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "ByteDance-Seed/SeedVR2-7B"
SEEDVR_REPO_PATH = "/tmp/seedvr_repo"

VID_IN_CHANNELS = 33
VID_OUT_CHANNELS = 16
TXT_IN_DIM = 5120
LATENT_T = 4
LATENT_H = 8
LATENT_W = 8
TXT_SEQ_LEN = 77


def _flash_attn_varlen_stub(q, k, v, *args, **kwargs):
    """CPU stand-in for flash_attn_varlen_func (returns zeros matching q shape)."""
    return torch.zeros_like(q)


def _ensure_seedvr_importable():
    """Clone the SeedVR GitHub repo for model code and make it importable."""
    if not os.path.isdir(SEEDVR_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/ByteDance-Seed/SeedVR.git",
                SEEDVR_REPO_PATH,
            ]
        )

    if SEEDVR_REPO_PATH not in sys.path:
        sys.path.insert(0, SEEDVR_REPO_PATH)

    if "flash_attn" not in sys.modules:
        flash_attn_mod = types.ModuleType("flash_attn")
        flash_attn_mod.flash_attn_varlen_func = _flash_attn_varlen_stub
        sys.modules["flash_attn"] = flash_attn_mod


class ModelVariant(StrEnum):
    """Available SeedVR2 model variants."""

    SEEDVR2_7B = "7B"
    SEEDVR2_7B_SHARP = "7B_Sharp"


class ModelLoader(ForgeModel):
    """ByteDance SeedVR2-7B video restoration model loader."""

    _VARIANTS = {
        ModelVariant.SEEDVR2_7B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.SEEDVR2_7B_SHARP: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEEDVR2_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._weights_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SeedVR2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_weights_path(self):
        """Download SeedVR2 weights from HuggingFace and return local path."""
        if self._weights_path is None:
            self._weights_path = snapshot_download(repo_id=REPO_ID)
        return self._weights_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeedVR2-7B NaDiT model."""
        _ensure_seedvr_importable()

        from models.dit.nadit import NaDiT

        num_layers = 36
        vid_dim = 3072
        model = NaDiT(
            vid_in_channels=VID_IN_CHANNELS,
            vid_out_channels=VID_OUT_CHANNELS,
            vid_dim=vid_dim,
            txt_in_dim=TXT_IN_DIM,
            txt_dim=vid_dim,
            emb_dim=6 * vid_dim,
            heads=24,
            head_dim=128,
            expand_ratio=4,
            norm="rms",
            norm_eps=1e-5,
            ada="single",
            qk_bias=False,
            qk_rope=True,
            qk_norm="rms",
            patch_size=[1, 2, 2],
            num_layers=num_layers,
            shared_mlp=False,
            shared_qkv=False,
            mlp_type="normal",
            block_type=num_layers * ["mmdit_sr"],
            window=num_layers * [(4, 3, 3)],
            window_method=(num_layers // 2)
            * ["720pwin_by_size_bysize", "720pswin_by_size_bysize"],
        )

        weights_path = self._get_weights_path()
        if self._variant == ModelVariant.SEEDVR2_7B_SHARP:
            ckpt_path = f"{weights_path}/seedvr2_ema_7b_sharp.pth"
        else:
            ckpt_path = f"{weights_path}/seedvr2_ema_7b.pth"

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Create synthetic inputs matching the NaDiT forward signature.

        NaDiT uses packed-sequence format:
        - vid: (total_tokens, channels) flattened spatiotemporal tokens
        - txt: (txt_tokens, txt_dim) text embeddings
        - vid_shape: (batch, 3) with [T, H, W] per video
        - txt_shape: (batch, 1) with text length per item
        - timestep: (batch,) diffusion timestep
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        total_vid_tokens = LATENT_T * LATENT_H * LATENT_W
        vid = torch.randn(total_vid_tokens, VID_IN_CHANNELS, dtype=dtype)
        txt = torch.randn(TXT_SEQ_LEN, TXT_IN_DIM, dtype=dtype)
        vid_shape = torch.tensor([[LATENT_T, LATENT_H, LATENT_W]], dtype=torch.long)
        txt_shape = torch.tensor([[TXT_SEQ_LEN]], dtype=torch.long)
        timestep = torch.tensor([1.0], dtype=dtype)

        return {
            "vid": vid,
            "txt": txt,
            "vid_shape": vid_shape,
            "txt_shape": txt_shape,
            "timestep": timestep,
        }
