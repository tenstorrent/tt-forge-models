# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diffusion Forcing Transformer (DFoT) model loader implementation.

DFoT is a video diffusion model that generates videos conditioned on context frames.
The RE10K variant uses a UViT3DPose backbone with camera-pose conditioning via
ray-encoded Plücker coordinates.

Reference: https://github.com/kwsong0113/diffusion-forcing-transformer
HuggingFace: https://huggingface.co/kiwhansong/DFoT
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]

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
from .src.model import UViT3DPoseConfig, load_uvit3d_pose_from_checkpoint

HF_REPO_ID = "kiwhansong/DFoT"


class ModelVariant(StrEnum):
    """Available DFoT model variants."""

    DFOT_RE10K = "RE10K"
    DFOT_K600 = "K600"
    DFOT_MCRAFT = "MCRAFT"


class ModelLoader(ForgeModel):
    """DFoT (Diffusion Forcing Transformer) model loader.

    Loads the UViT3DPose backbone from pretrained PyTorch Lightning checkpoints
    hosted on HuggingFace. Each variant corresponds to a different training
    dataset:
      - RE10K: RealEstate10K (camera-pose conditioned video generation)
      - K600: Kinetics-600 (unconditional video generation)
      - MCRAFT: Minecraft (unconditional video generation)
    """

    _VARIANTS = {
        ModelVariant.DFOT_RE10K: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_RE10K.ckpt",
        ),
        ModelVariant.DFOT_K600: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_K600.ckpt",
        ),
        ModelVariant.DFOT_MCRAFT: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_MCRAFT.ckpt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DFOT_RE10K

    _DEFAULT_CFG = UViT3DPoseConfig(
        channels=(128, 256, 576, 1152),
        emb_channels=1024,
        patch_size=2,
        block_types=(
            "ResBlock",
            "ResBlock",
            "TransformerBlock",
            "TransformerBlock",
        ),
        block_dropouts=(0.0, 0.0, 0.1, 0.1),
        num_updown_blocks=(3, 3, 6),
        num_mid_blocks=20,
        num_heads=9,
        in_channels=3,
        external_cond_dim=180,
        resolution=64,
        temporal_length=2,
    )

    DEFAULT_NUM_FRAMES = 2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DFoT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        ckpt_filename = self._variant_config.pretrained_model_name
        ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ckpt_filename)
        model = load_uvit3d_pose_from_checkpoint(ckpt_path, self._DEFAULT_CFG)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, num_frames=None, **kwargs):
        cfg = self._DEFAULT_CFG
        T = num_frames or self.DEFAULT_NUM_FRAMES
        dtype = dtype_override or torch.float32

        x = torch.randn(
            1, T, cfg.in_channels, cfg.resolution, cfg.resolution, dtype=dtype
        )
        noise_levels = torch.randn(1, T, dtype=dtype)
        external_cond = torch.randn(
            1, T, cfg.external_cond_dim, cfg.resolution, cfg.resolution, dtype=dtype
        )

        return [x, noise_levels, external_cond]
