# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .aggregator import Aggregator
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    """Visual Geometry Grounded Transformer.

    Port of facebookresearch/vggt with camera, depth, and point heads. The
    tracking head is intentionally omitted; weights for it are silently
    skipped when loading via ``PyTorchModelHubMixin.from_pretrained`` (which
    loads non-strict by default).
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )

    def forward(self, images: torch.Tensor):
        """Forward pass.

        Args:
            images: Input tensor of shape ``[S, 3, H, W]`` or ``[B, S, 3, H, W]``
                with values in ``[0, 1]``.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        images = images.to(next(self.parameters()).dtype)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.point_head is not None:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        if not self.training:
            predictions["images"] = images

        return predictions
