# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simplified BEVFusion model implementation.

Reference: https://github.com/mit-han-lab/bevfusion
BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.

Note: The original BEVFusion requires CUDA-specific ops (spconv, mmcv, mmdet3d).
This is a simplified PyTorch-only reimplementation that captures the core architecture:
  - Camera branch: image backbone + BEV view transform
  - LiDAR branch: voxel feature extraction + BEV projection
  - Fusion: BEV feature fusion + detection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU building block."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CameraBackbone(nn.Module):
    """Simplified camera image backbone (ResNet-like stem + neck).

    Produces a BEV feature map from N_cam × C × H × W camera images.
    Uses a lightweight encoder + view transform (lift-splat-style projection).
    """

    def __init__(self, in_channels=3, bev_channels=64, bev_h=128, bev_w=128):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Image feature extractor
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 2, 1),
            ConvBNReLU(32, 64, 3, 2, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
        )

        # Depth prediction (simplified: predict BEV occupancy weights)
        self.depth_pred = nn.Conv2d(128, 64, 1)

        # Context feature
        self.context = nn.Conv2d(128, bev_channels, 1)

        # BEV pooling projection
        self.bev_proj = nn.Sequential(
            ConvBNReLU(bev_channels, bev_channels, 3, 1, 1),
            nn.AdaptiveAvgPool2d((bev_h, bev_w)),
        )

    def forward(self, imgs):
        """
        Args:
            imgs: (B, N_cam, C, H, W)
        Returns:
            bev_feats: (B, bev_channels, bev_h, bev_w)
        """
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)

        feats = self.encoder(imgs_flat)  # (B*N, 128, H/8, W/8)
        context = self.context(feats)  # (B*N, bev_channels, H/8, W/8)

        # Aggregate over cameras by mean pooling then project to BEV
        _, bc, fh, fw = context.shape
        context = context.view(B, N, bc, fh, fw).mean(dim=1)  # (B, bc, fh, fw)
        bev_feats = self.bev_proj(context)
        return bev_feats


class LiDARBranch(nn.Module):
    """Simplified LiDAR voxel backbone.

    Takes pre-voxelized features (pseudo BEV grid) and produces BEV features.
    Avoids spconv by using standard 3D→2D convolutions.
    """

    def __init__(self, in_channels=64, bev_channels=64, bev_h=128, bev_w=128):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, 1, 1),
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, bev_channels, 3, 1, 1),
        )
        self.bev_adapt = nn.AdaptiveAvgPool2d((bev_h, bev_w))

    def forward(self, voxel_bev):
        """
        Args:
            voxel_bev: (B, in_channels, H, W) — pre-computed BEV voxel features
        Returns:
            bev_feats: (B, bev_channels, bev_h, bev_w)
        """
        feats = self.backbone(voxel_bev)
        return self.bev_adapt(feats)


class BEVFusionNeck(nn.Module):
    """BEV feature fusion neck.

    Concatenates camera + LiDAR BEV features and applies convolutional fusion.
    """

    def __init__(self, cam_channels=64, lidar_channels=64, out_channels=128):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNReLU(cam_channels + lidar_channels, out_channels, 3, 1, 1),
            ConvBNReLU(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, cam_bev, lidar_bev):
        """
        Args:
            cam_bev:   (B, cam_channels, H, W)
            lidar_bev: (B, lidar_channels, H, W)
        Returns:
            fused: (B, out_channels, H, W)
        """
        return self.fuse(torch.cat([cam_bev, lidar_bev], dim=1))


class DetectionHead(nn.Module):
    """Simple center-based detection head.

    Produces class heatmaps + regression outputs on BEV grid.
    """

    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        self.heatmap = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes, 1),
        )
        self.regression = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, 1, 1),
            nn.Conv2d(64, 8, 1),  # (x, y, z, w, l, h, sin_yaw, cos_yaw)
        )

    def forward(self, bev_feats):
        return {
            "heatmap": self.heatmap(bev_feats),
            "regression": self.regression(bev_feats),
        }


class BEVFusionCameraOnlyModel(nn.Module):
    """Camera-only variant of BEVFusion for 3D object detection.

    Uses only the camera branch (no LiDAR). Takes multi-view images and
    produces BEV heatmap + regression outputs.
    Reference: https://github.com/mit-han-lab/bevfusion
    """

    def __init__(
        self,
        num_cameras=6,
        img_channels=3,
        bev_channels=64,
        bev_h=128,
        bev_w=128,
        num_classes=10,
    ):
        super().__init__()
        self.camera_branch = CameraBackbone(img_channels, bev_channels, bev_h, bev_w)
        self.head = DetectionHead(bev_channels, num_classes)

    def forward(self, imgs):
        """
        Args:
            imgs: (B, N_cam, 3, H, W) — camera images from all cameras
        Returns:
            dict with 'heatmap' (B, num_classes, H_bev, W_bev)
                  and 'regression' (B, 8, H_bev, W_bev)
        """
        cam_bev = self.camera_branch(imgs)
        return self.head(cam_bev)


class BEVFusionModel(nn.Module):
    """Simplified BEVFusion model for 3D object detection.

    Fuses camera images and LiDAR voxel features in Bird's Eye View (BEV) space.
    Reference: https://github.com/mit-han-lab/bevfusion
    """

    def __init__(
        self,
        num_cameras=6,
        img_channels=3,
        lidar_in_channels=64,
        bev_channels=64,
        bev_h=128,
        bev_w=128,
        num_classes=10,
    ):
        super().__init__()
        self.camera_branch = CameraBackbone(img_channels, bev_channels, bev_h, bev_w)
        self.lidar_branch = LiDARBranch(lidar_in_channels, bev_channels, bev_h, bev_w)
        self.neck = BEVFusionNeck(bev_channels, bev_channels, bev_channels * 2)
        self.head = DetectionHead(bev_channels * 2, num_classes)

    def forward(self, imgs, voxel_bev):
        """
        Args:
            imgs:      (B, N_cam, 3, H, W) — camera images from all cameras
            voxel_bev: (B, lidar_in_channels, H_bev, W_bev) — LiDAR voxel BEV projection
        Returns:
            dict with 'heatmap' (B, num_classes, H_bev, W_bev)
                  and 'regression' (B, 8, H_bev, W_bev)
        """
        cam_bev = self.camera_branch(imgs)
        lidar_bev = self.lidar_branch(voxel_bev)
        fused = self.neck(cam_bev, lidar_bev)
        return self.head(fused)
