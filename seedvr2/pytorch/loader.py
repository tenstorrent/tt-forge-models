# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance SeedVR2-7B video restoration model loader implementation.

SeedVR2 is a one-step diffusion-based video restoration model using a NaDiT
(Neighborhood Attention Diffusion Transformer) architecture. It takes degraded
low-quality video frames and produces high-quality restored output.

Requires the SeedVR GitHub repo (https://github.com/ByteDance-Seed/SeedVR)
to be cloned for model architecture code.

Available variants:
- SEEDVR2_7B: Standard 7B parameter model (seedvr2_ema_7b.pth)
- SEEDVR2_7B_SHARP: Sharper variant (seedvr2_ema_7b_sharp.pth)
"""

import os
import sys
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
SEEDVR_CODE_REPO_URL = "https://github.com/ByteDance-Seed/SeedVR.git"
SEEDVR_CODE_REPO_PATH = "/tmp/seedvr_repo"

LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 4
TEXT_SEQ_LEN = 77
TEXT_DIM = 5120


def _flash_attn_varlen_func_shim(
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
    window_size=(-1, -1),
    return_attn_probs=False,
    deterministic=False,
):
    """Pure-PyTorch replacement for flash_attn_varlen_func."""
    import torch.nn.functional as F

    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(batch_size):
        sq_start, sq_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
        sk_start, sk_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
        qi = q[sq_start:sq_end].unsqueeze(0).transpose(1, 2)
        ki = k[sk_start:sk_end].unsqueeze(0).transpose(1, 2)
        vi = v[sk_start:sk_end].unsqueeze(0).transpose(1, 2)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, is_causal=causal, scale=softmax_scale
        )
        outputs.append(oi.transpose(1, 2).squeeze(0))
    return torch.cat(outputs, dim=0)


def _ensure_flash_attn_available():
    """Provide a shim flash_attn module when the real one is unavailable."""
    if "flash_attn" not in sys.modules:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            import types

            mod = types.ModuleType("flash_attn")
            mod.flash_attn_varlen_func = _flash_attn_varlen_func_shim
            sys.modules["flash_attn"] = mod


def _ensure_seedvr_importable():
    """Clone the SeedVR GitHub repo and make it importable."""
    if not os.path.isdir(SEEDVR_CODE_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                SEEDVR_CODE_REPO_URL,
                SEEDVR_CODE_REPO_PATH,
            ]
        )

    _ensure_flash_attn_available()

    if SEEDVR_CODE_REPO_PATH not in sys.path:
        sys.path.insert(0, SEEDVR_CODE_REPO_PATH)


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
        self._hf_repo_path = None

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

    def _get_hf_repo_path(self):
        if self._hf_repo_path is None:
            self._hf_repo_path = snapshot_download(repo_id=REPO_ID)
        return self._hf_repo_path

    def _create_nadit_model(self):
        """Instantiate the NaDiT model from the SeedVR codebase using config."""
        _ensure_seedvr_importable()

        from omegaconf import OmegaConf

        OmegaConf.clear_resolver("eval")
        OmegaConf.register_new_resolver("eval", eval)

        config_path = os.path.join(SEEDVR_CODE_REPO_PATH, "configs_7b", "main.yaml")
        config = OmegaConf.load(config_path)
        dit_model_config = OmegaConf.to_object(config.dit.model)

        dit_model_config.pop("__object__")

        # Replace fused norm types (require NVIDIA apex) with standard equivalents
        if dit_model_config.get("norm") == "fusedrms":
            dit_model_config["norm"] = "rms"
        if dit_model_config.get("qk_norm") == "fusedrms":
            dit_model_config["qk_norm"] = "rms"

        from models.dit.nadit import NaDiT

        return NaDiT(**dit_model_config)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeedVR2-7B NaDiT model."""
        model = self._create_nadit_model()

        if not os.environ.get("TT_RANDOM_WEIGHTS"):
            hf_repo_path = self._get_hf_repo_path()

            if self._variant == ModelVariant.SEEDVR2_7B_SHARP:
                ckpt_path = f"{hf_repo_path}/seedvr2_ema_7b_sharp.pth"
            else:
                ckpt_path = f"{hf_repo_path}/seedvr2_ema_7b.pth"

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load synthetic inputs for the SeedVR2 NaDiT model.

        NaDiT uses a native-resolution format where tensors are flattened:
        - vid: (total_elements, channels) packed video latents
        - txt: (total_tokens, text_dim) text embeddings
        - vid_shape: (batch, 3) spatial-temporal dimensions per sample
        - txt_shape: (batch, 1) text length per sample
        - timestep: (batch,) diffusion timestep
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        total_elements = LATENT_DEPTH * LATENT_HEIGHT * LATENT_WIDTH
        vid_in_channels = 33

        vid = torch.randn(total_elements, vid_in_channels, dtype=dtype)
        txt = torch.randn(TEXT_SEQ_LEN, TEXT_DIM, dtype=dtype)
        vid_shape = torch.tensor(
            [[LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH]], dtype=torch.long
        )
        txt_shape = torch.tensor([[TEXT_SEQ_LEN]], dtype=torch.long)
        timestep = torch.tensor([1.0], dtype=dtype)

        return {
            "vid": vid,
            "txt": txt,
            "vid_shape": vid_shape,
            "txt_shape": txt_shape,
            "timestep": timestep,
        }
