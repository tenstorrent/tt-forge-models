# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFusion model loader implementation.

Reference: https://github.com/mit-han-lab/bevfusion
BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.

Note: Original BEVFusion requires CUDA-specific ops (spconv, mmcv, mmdet3d).
This loader uses a simplified PyTorch-only implementation.
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
from .src.model import BEVFusionModel, BEVFusionCameraOnlyModel


@dataclass
class BEVFusionConfig(ModelConfig):
    """Configuration specific to BEVFusion variants."""

    num_cameras: int = 6
    img_h: int = 256
    img_w: int = 704
    bev_h: int = 128
    bev_w: int = 128
    lidar_in_channels: int = 64
    num_classes: int = 10


class ModelVariant(StrEnum):
    """Available BEVFusion model variants."""

    BEVFUSION_CAMERA_LIDAR = "BEVFusion_Camera_Lidar"
    BEVFUSION_CAMERA_ONLY = "BEVFusion_Camera_Only"


class ModelLoader(ForgeModel):
    """BEVFusion model loader for multi-sensor 3D object detection."""

    _VARIANTS = {
        ModelVariant.BEVFUSION_CAMERA_LIDAR: BEVFusionConfig(
            pretrained_model_name="bevfusion_camera_lidar",
            num_cameras=6,
            img_h=256,
            img_w=704,
            bev_h=128,
            bev_w=128,
            lidar_in_channels=64,
            num_classes=10,
        ),
        ModelVariant.BEVFUSION_CAMERA_ONLY: BEVFusionConfig(
            pretrained_model_name="bevfusion_camera_only",
            num_cameras=6,
            img_h=256,
            img_w=704,
            bev_h=128,
            bev_w=128,
            lidar_in_channels=64,
            num_classes=10,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEVFUSION_CAMERA_LIDAR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BEVFusion",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MULTIVIEW_3D_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BEVFusion model instance.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: BEVFusionModel or BEVFusionCameraOnlyModel instance.
        """
        cfg = self._variant_config
        if self._variant == ModelVariant.BEVFUSION_CAMERA_ONLY:
            model = BEVFusionCameraOnlyModel(
                num_cameras=cfg.num_cameras,
                img_channels=3,
                bev_channels=64,
                bev_h=cfg.bev_h,
                bev_w=cfg.bev_w,
                num_classes=cfg.num_classes,
            )
        else:
            model = BEVFusionModel(
                num_cameras=cfg.num_cameras,
                img_channels=3,
                lidar_in_channels=cfg.lidar_in_channels,
                bev_channels=64,
                bev_h=cfg.bev_h,
                bev_w=cfg.bev_w,
                num_classes=cfg.num_classes,
            )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for BEVFusion.

        For Camera_Lidar variant returns (imgs, voxel_bev).
        For Camera_Only variant returns (imgs,).

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            tuple of input tensors.
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        imgs = torch.randn(
            batch_size, cfg.num_cameras, 3, cfg.img_h, cfg.img_w, dtype=dtype
        )
        if self._variant == ModelVariant.BEVFUSION_CAMERA_ONLY:
            return (imgs,)

        voxel_bev = torch.randn(
            batch_size, cfg.lidar_in_channels, cfg.bev_h, cfg.bev_w, dtype=dtype
        )
        return imgs, voxel_bev
