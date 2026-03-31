# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
3D UNet model loader for volumetric image segmentation.

Reference: https://github.com/tenstorrent/Toyota (Toyota PR architecture)

GroupNorm-based 3D UNet architecture for medical image segmentation.
Uses 3D convolutions with skip connections between encoder and decoder.
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
from .src.model import UNet3D


@dataclass
class UNet3DConfig(ModelConfig):
    """Configuration for 3D UNet variants."""

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    num_groups: int = 8
    # Input volume spatial dimensions
    depth: int = 32
    height: int = 32
    width: int = 64


class ModelVariant(StrEnum):
    """Available 3D UNet variants."""

    UNET3D_SMALL = "UNet3D_Small"
    UNET3D_LARGE = "UNet3D_Large"


class ModelLoader(ForgeModel):
    """3D UNet model loader for volumetric segmentation."""

    _VARIANTS = {
        ModelVariant.UNET3D_SMALL: UNet3DConfig(
            pretrained_model_name="unet_3d_small",
            in_channels=1,
            out_channels=1,
            base_channels=16,
            num_groups=8,
            depth=32,
            height=32,
            width=64,
        ),
        ModelVariant.UNET3D_LARGE: UNet3DConfig(
            pretrained_model_name="unet_3d_large",
            in_channels=32,
            out_channels=32,
            base_channels=32,
            num_groups=8,
            depth=16,
            height=64,
            width=16,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNET3D_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UNet3D",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the 3D UNet model.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: UNet3D instance.
        """
        cfg = self._variant_config
        model = UNet3D(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            base_channels=cfg.base_channels,
            num_groups=cfg.num_groups,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample volumetric inputs for 3D UNet.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            tuple: (input_tensor,) where input_tensor is (B, in_channels, D, H, W)
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        volume = torch.randn(
            batch_size, cfg.in_channels, cfg.depth, cfg.height, cfg.width,
            dtype=dtype
        )
        return (volume,)
