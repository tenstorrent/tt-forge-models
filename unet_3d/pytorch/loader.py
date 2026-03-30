# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet 3D model loader implementation
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


@dataclass
class UNet3DConfig(ModelConfig):
    """Configuration specific to UNet 3D models"""

    in_channels: int
    out_channels: int
    base_channels: int
    num_groups: int
    depth: int
    height: int
    width: int


class ModelVariant(StrEnum):
    """Available UNet 3D model variants."""

    SMALL = "UNet3D_Small"
    LARGE = "UNet3D_Large"


class ModelLoader(ForgeModel):
    """UNet 3D model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMALL: UNet3DConfig(
            pretrained_model_name="unet_3d_small",
            in_channels=1,
            out_channels=1,
            base_channels=16,
            num_groups=8,
            depth=32,
            height=32,
            width=64,
        ),
        ModelVariant.LARGE: UNet3DConfig(
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

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """
        Initialize the UNet 3D model loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config: UNet3DConfig = self._variant_config

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="UNet 3D",
            variant=variant,
            group=ModelGroup.GENERALITY,
            source=ModelSource.GITHUB,
            task=ModelTask.CV_IMAGE_SEG,
            framework=Framework.TORCH,
        )

    def load_model(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> torch.nn.Module:
        """
        Load the UNet 3D model.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).

        Returns:
            torch.nn.Module: Loaded UNet 3D model
        """
        from .src.model import UNet3D

        model = UNet3D(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            num_groups=self.config.num_groups,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(
        self, batch_size: int = 1, dtype_override: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Generate random input tensor for the model.

        Args:
            batch_size: Batch size for the input tensor. Default: 1
            dtype_override: Optional torch.dtype override (default: float32).

        Returns:
            torch.Tensor: Random input tensor
        """
        inputs = torch.randn(
            batch_size,
            self.config.in_channels,
            self.config.depth,
            self.config.height,
            self.config.width,
            dtype=torch.float32,
        )

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
