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
import subprocess
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

HF_REPO_ID = "ByteDance-Seed/SeedVR2-7B"
GITHUB_REPO_URL = "https://github.com/ByteDance-Seed/SeedVR.git"
CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "seedvr2_src",
)

LATENT_CHANNELS = 16
CONDITION_CHANNELS = 16
MASK_CHANNELS = 1
INPUT_CHANNELS = LATENT_CHANNELS + CONDITION_CHANNELS + MASK_CHANNELS  # 33
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 4


def _ensure_repo_cloned():
    if not os.path.isdir(os.path.join(CACHE_DIR, ".git")):
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO_URL, CACHE_DIR],
            check=True,
            capture_output=True,
        )
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)


def _flash_attn_varlen_stub(q, k, v, *args, **kwargs):
    return torch.zeros_like(q)


def _mock_flash_attn():
    if "flash_attn" not in sys.modules:
        mock = types.ModuleType("flash_attn")
        mock.flash_attn_varlen_func = _flash_attn_varlen_stub
        sys.modules["flash_attn"] = mock


class ModelVariant(StrEnum):
    SEEDVR2_7B = "7B"
    SEEDVR2_7B_SHARP = "7B_Sharp"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.SEEDVR2_7B: ModelConfig(
            pretrained_model_name=HF_REPO_ID,
        ),
        ModelVariant.SEEDVR2_7B_SHARP: ModelConfig(
            pretrained_model_name=HF_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEEDVR2_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._hf_path = None

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

    def _get_hf_path(self):
        if self._hf_path is None:
            self._hf_path = snapshot_download(repo_id=HF_REPO_ID)
        return self._hf_path

    def load_model(self, *, dtype_override=None, **kwargs):
        from omegaconf import OmegaConf

        _mock_flash_attn()
        _ensure_repo_cloned()

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        config = OmegaConf.load(os.path.join(CACHE_DIR, "configs_7b", "main.yaml"))
        dit_params = OmegaConf.to_container(config.dit.model, resolve=True)
        dit_params.pop("__object__", None)
        # FusedRMSNorm requires NVIDIA Apex (CUDA-only); use diffusers RMSNorm
        if dit_params.get("norm") == "fusedrms":
            dit_params["norm"] = "rms"
        if dit_params.get("qk_norm") == "fusedrms":
            dit_params["qk_norm"] = "rms"

        from models.dit.nadit import NaDiT

        model = NaDiT(**dit_params)

        hf_path = self._get_hf_path()
        if self._variant == ModelVariant.SEEDVR2_7B_SHARP:
            ckpt_path = os.path.join(hf_path, "seedvr2_ema_7b_sharp.pth")
        else:
            ckpt_path = os.path.join(hf_path, "seedvr2_ema_7b.pth")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.float32
        txt_len = 77
        txt_dim = 5120

        vid = torch.randn(
            LATENT_DEPTH * LATENT_HEIGHT * LATENT_WIDTH,
            INPUT_CHANNELS,
            dtype=dtype,
        )
        vid_shape = torch.tensor(
            [[LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]], dtype=torch.long
        )
        txt = torch.randn(txt_len, txt_dim, dtype=dtype)
        txt_shape = torch.tensor([[txt_len]], dtype=torch.long)
        timestep = torch.tensor([1.0], dtype=dtype)

        return {
            "vid": vid,
            "txt": txt,
            "vid_shape": vid_shape,
            "txt_shape": txt_shape,
            "timestep": timestep,
        }
