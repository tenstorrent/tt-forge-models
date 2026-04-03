# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeRF (Neural Radiance Fields) model loader.

Reference: https://github.com/Yubel426/NeRF-3DGS-at-CVPR-2024
Original NeRF: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
(Mildenhall et al., ECCV 2020)

This loader provides the core NeRF MLP inference model.
Input: rays (3D positions + view directions along the ray)
Output: RGB color + density at sampled positions (for volume rendering)
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model import NeRFMLP


@dataclass
class NeRFConfig(ModelConfig):
    """Configuration for NeRF variants."""

    pos_freq: int = 10
    dir_freq: int = 4
    hidden_dim: int = 256
    # Number of points sampled per ray
    num_ray_samples: int = 64
    # Batch of rays (H*W or subset)
    num_rays: int = 1024


class ModelVariant(StrEnum):
    """Available NeRF model variants."""

    NERF_VANILLA = "NeRF_Vanilla"
    NERF_COARSE = "NeRF_Coarse"


class ModelLoader(ForgeModel):
    """NeRF MLP model loader for novel view synthesis."""

    _VARIANTS = {
        ModelVariant.NERF_VANILLA: NeRFConfig(
            pretrained_model_name="nerf_vanilla",
            pos_freq=10,
            dir_freq=4,
            hidden_dim=256,
            num_ray_samples=64,
            num_rays=1024,
        ),
        ModelVariant.NERF_COARSE: NeRFConfig(
            pretrained_model_name="nerf_coarse",
            pos_freq=6,
            dir_freq=4,
            hidden_dim=128,
            num_ray_samples=32,
            num_rays=1024,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NERF_VANILLA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NeRF",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NeRF MLP model.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: NeRFMLP instance.
        """
        cfg = self._variant_config
        model = NeRFMLP(
            pos_freq=cfg.pos_freq,
            dir_freq=cfg.dir_freq,
            hidden_dim=cfg.hidden_dim,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample ray inputs for NeRF.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (number of ray batches, default: 1).

        Returns:
            tuple: (positions, directions)
              - positions:  (B * num_rays * num_samples, 3) 3D sample positions
              - directions: (B * num_rays * num_samples, 3) normalized view directions
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        total_pts = batch_size * cfg.num_rays * cfg.num_ray_samples

        positions = torch.randn(total_pts, 3, dtype=dtype)
        raw_dirs = torch.randn(total_pts, 3, dtype=dtype)
        directions = torch.nn.functional.normalize(raw_dirs, dim=-1)

        return positions, directions
