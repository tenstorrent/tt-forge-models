# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance SeedVR2 video restoration model loader implementation.

SeedVR2 is a one-step diffusion-based video restoration model using a NaDiT
(Neighborhood Attention Diffusion Transformer) architecture. It takes degraded
low-quality video frames and produces high-quality restored output.

Available variants:
- SEEDVR2_3B: Standard 3B parameter model (seedvr2_ema_3b.pth)
- SEEDVR2_7B: Standard 7B parameter model (seedvr2_ema_7b.pth)
- SEEDVR2_7B_SHARP: Sharper 7B variant (seedvr2_ema_7b_sharp.pth)
"""

import os
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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

SEEDVR_REPO_PATH = "/tmp/seedvr_repo"

# NaDiT model input dimensions for testing
# The model operates in latent space: patch_size [1, 2, 2],
# VAE compression 4x temporal / 8x spatial, 16 latent channels
LATENT_CHANNELS = 16
CONDITION_CHANNELS = 16
MASK_CHANNELS = 1
INPUT_CHANNELS = LATENT_CHANNELS + CONDITION_CHANNELS + MASK_CHANNELS  # 33
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 4  # temporal frames in latent space


def _flash_attn_varlen_func_fallback(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    """CPU fallback for flash_attn_varlen_func using standard PyTorch attention."""
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(batch_size):
        qs = cu_seqlens_q[i].item()
        qe = cu_seqlens_q[i + 1].item()
        ks = cu_seqlens_k[i].item()
        ke = cu_seqlens_k[i + 1].item()
        # q, k, v: (seq_len, num_heads, head_dim) -> (1, num_heads, seq_len, head_dim)
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)
        scale = softmax_scale or (q_i.shape[-1] ** -0.5)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            scale=scale,
            dropout_p=dropout_p if not causal else 0.0,
            is_causal=causal,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _inject_stubs():
    """Inject CPU-compatible stubs for flash_attn and apex before importing SeedVR modules."""
    if "flash_attn" not in sys.modules:
        flash_attn_mod = types.ModuleType("flash_attn")
        flash_attn_mod.flash_attn_varlen_func = _flash_attn_varlen_func_fallback
        sys.modules["flash_attn"] = flash_attn_mod

    if "apex" not in sys.modules:
        apex_mod = types.ModuleType("apex")
        apex_norm_mod = types.ModuleType("apex.normalization")

        class FusedRMSNorm(nn.RMSNorm):
            def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
                super().__init__(
                    normalized_shape, eps=eps, elementwise_affine=elementwise_affine
                )

        class FusedLayerNorm(nn.LayerNorm):
            def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-5):
                super().__init__(
                    normalized_shape, eps=eps, elementwise_affine=elementwise_affine
                )

        apex_norm_mod.FusedRMSNorm = FusedRMSNorm
        apex_norm_mod.FusedLayerNorm = FusedLayerNorm
        apex_mod.normalization = apex_norm_mod
        sys.modules["apex"] = apex_mod
        sys.modules["apex.normalization"] = apex_norm_mod


def _ensure_seedvr_importable():
    """Clone the SeedVR GitHub repo for model code and configs, inject stubs."""
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

    _inject_stubs()

    if SEEDVR_REPO_PATH not in sys.path:
        # Remove any stale 'utils' module that might shadow the SeedVR packages
        for mod in list(sys.modules.keys()):
            if mod == "utils" or mod.startswith("utils."):
                del sys.modules[mod]
        sys.path.insert(0, SEEDVR_REPO_PATH)


class ModelVariant(StrEnum):
    """Available SeedVR2 model variants."""

    SEEDVR2_3B = "3B"
    SEEDVR2_7B = "7B"
    SEEDVR2_7B_SHARP = "7B_Sharp"


_VARIANT_CHECKPOINTS = {
    ModelVariant.SEEDVR2_3B: ("configs_3b/main.yaml", "seedvr2_ema_3b.pth"),
    ModelVariant.SEEDVR2_7B: ("configs_7b/main.yaml", "seedvr2_ema_7b.pth"),
    ModelVariant.SEEDVR2_7B_SHARP: ("configs_7b/main.yaml", "seedvr2_ema_7b_sharp.pth"),
}


class ModelLoader(ForgeModel):
    """ByteDance SeedVR2 video restoration model loader."""

    _VARIANTS = {
        ModelVariant.SEEDVR2_3B: ModelConfig(
            pretrained_model_name="ByteDance-Seed/SeedVR2-3B",
        ),
        ModelVariant.SEEDVR2_7B: ModelConfig(
            pretrained_model_name="ByteDance-Seed/SeedVR2-7B",
        ),
        ModelVariant.SEEDVR2_7B_SHARP: ModelConfig(
            pretrained_model_name="ByteDance-Seed/SeedVR2-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEEDVR2_7B

    _CONFIG_DIRS = {
        ModelVariant.SEEDVR2_3B: "configs_3b",
        ModelVariant.SEEDVR2_7B: "configs_7b",
        ModelVariant.SEEDVR2_7B_SHARP: "configs_7b",
    }

    _CKPT_FILES = {
        ModelVariant.SEEDVR2_3B: "seedvr2_ema_3b.pth",
        ModelVariant.SEEDVR2_7B: "seedvr2_ema_7b.pth",
        ModelVariant.SEEDVR2_7B_SHARP: "seedvr2_ema_7b_sharp.pth",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._repo_path = None

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

    def _get_repo_path(self):
        """Download the SeedVR2 weights from HuggingFace and return the local path."""
        if self._repo_path is None:
            self._repo_path = snapshot_download(
                repo_id=self._variant_config.pretrained_model_name
            )
        return self._repo_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeedVR2 NaDiT model.

        Clones the SeedVR GitHub repo for model code/configs and downloads
        the weights from HuggingFace.

        Returns:
            torch.nn.Module: The NaDiT diffusion transformer model.
        """
        _ensure_seedvr_importable()
        repo_path = self._get_repo_path()

        from common.config import create_object, load_config

        config_rel, ckpt_rel = _VARIANT_CHECKPOINTS[self._variant]

        # load_config resolves __inherit__ with relative paths; run from repo root
        old_cwd = os.getcwd()
        try:
            os.chdir(SEEDVR_REPO_PATH)
            config = load_config(config_rel)
        finally:
            os.chdir(old_cwd)

        model = create_object(config.dit.model)

        ckpt_path = f"{repo_path}/{ckpt_rel}"
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load latent-space inputs for the SeedVR2 NaDiT model.

        NaDiT.forward expects packed sequence format:
          vid, txt, vid_shape, txt_shape, timestep

        Returns:
            tuple: (vid, txt, vid_shape, txt_shape, timestep)
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        repo_path = self._get_repo_path()

        # vid: packed video latent (T*H*W, INPUT_CHANNELS=33)
        vid = torch.randn(
            LATENT_DEPTH * LATENT_HEIGHT * LATENT_WIDTH, INPUT_CHANNELS, dtype=dtype
        )
        # vid_shape: (batch=1, 3) = [[T, H, W]]
        vid_shape = torch.tensor(
            [[LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]], dtype=torch.long
        )

        # Load pre-computed text embeddings from HF repo (shape: [seq_len, 5120])
        pos_emb = torch.load(
            f"{repo_path}/pos_emb.pt", map_location="cpu", weights_only=True
        ).to(dtype=dtype)
        if pos_emb.dim() == 1:
            pos_emb = pos_emb.unsqueeze(0)

        # txt: packed text embedding (seq_len, txt_in_dim=5120)
        txt = pos_emb
        # txt_shape: (batch=1, 1) = [[seq_len]]
        txt_shape = torch.tensor([[txt.shape[0]]], dtype=torch.long)

        # Single-step diffusion timestep
        timestep = torch.tensor([1.0], dtype=dtype)

        return (vid, txt, vid_shape, txt_shape, timestep)
