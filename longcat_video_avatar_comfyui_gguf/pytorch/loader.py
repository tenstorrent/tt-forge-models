# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video-Avatar model loader implementation.

LongCat is an audio-driven character animation model that generates
expressive talking avatar videos. Uses the custom
LongCatVideoAvatarTransformer3DModel from the meituan-longcat/LongCat-Video
codebase, loaded from the official diffusers-format weights.

Repository:
- https://huggingface.co/meituan-longcat/LongCat-Video-Avatar
"""
import sys
import types

import torch
import torch.nn.functional as F
from typing import Optional

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

BASE_REPO = "meituan-longcat/LongCat-Video-Avatar"
LONGCAT_VIDEO_REPO = "https://github.com/meituan-longcat/LongCat-Video.git"


class ModelVariant(StrEnum):
    """Available LongCat-Video-Avatar model variants."""

    SINGLE_Q4_K_M = "Single_Q4_K_M"
    SINGLE_Q8_0 = "Single_Q8_0"


def _install_flash_attn_shim():
    """Install a fake flash_attn module that uses PyTorch SDPA."""
    if "flash_attn" in sys.modules:
        return

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **kwargs):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        batch = len(cu_seqlens_q) - 1
        outputs = []
        for i in range(batch):
            qs, qe = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            ks, ke = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
            ki = k[ks:ke].unsqueeze(0).transpose(1, 2)
            vi = v[ks:ke].unsqueeze(0).transpose(1, 2)
            out = F.scaled_dot_product_attention(qi, ki, vi)
            outputs.append(out.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)

    mod = types.ModuleType("flash_attn")
    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod


def _ensure_longcat_video():
    """Clone and add the LongCat-Video repo to sys.path if not importable."""
    _install_flash_attn_shim()

    try:
        import longcat_video  # noqa: F401

        return
    except ImportError:
        pass

    import subprocess
    import tempfile

    clone_dir = tempfile.mkdtemp(prefix="longcat_video_")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", LONGCAT_VIDEO_REPO, clone_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sys.path.insert(0, clone_dir)


class ModelLoader(ForgeModel):
    """LongCat-Video-Avatar model loader for audio-driven avatar animation."""

    _VARIANTS = {
        ModelVariant.SINGLE_Q4_K_M: ModelConfig(
            pretrained_model_name=BASE_REPO,
        ),
        ModelVariant.SINGLE_Q8_0: ModelConfig(
            pretrained_model_name=BASE_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SINGLE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LongCat-Video-Avatar-ComfyUI GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_longcat_video()
        from longcat_video.modules.avatar.longcat_video_dit_avatar import (
            LongCatVideoAvatarTransformer3DModel,
        )

        subfolder = "avatar_single"

        self.transformer = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            BASE_REPO,
            subfolder=subfolder,
        )

        for module in self.transformer.modules():
            if hasattr(module, "cp_split_hw") and module.cp_split_hw is None:
                module.cp_split_hw = [1, 1]

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        config = self.transformer.config

        num_channels = config.in_channels
        num_frames = 9
        height = 60
        width = 104
        vae_scale = config.vae_scale
        audio_window = config.audio_window
        audio_blocks = self.transformer.audio_proj.blocks
        audio_channels = self.transformer.audio_proj.channels
        audio_num_frames = 1 + (num_frames - 1) * vae_scale

        hidden_states = torch.randn(batch_size, num_channels, num_frames, height, width)

        timestep = torch.tensor([1.0]).expand(batch_size)

        encoder_hidden_states = torch.randn(batch_size, 1, 256, config.caption_channels)

        audio_embs = torch.randn(
            batch_size,
            audio_num_frames,
            audio_window,
            audio_blocks,
            audio_channels,
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_embs": audio_embs,
        }

        return inputs
