# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# source: https://github.com/tom-roddick/oft/blob/master/oft/model/oft.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils  # from .. import utils

EPSILON = 1e-6


class OFT(nn.Module):
    def __init__(self, channels, cell_size, grid_height, scale=1):
        super().__init__()

        y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
        y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
        self.register_buffer("y_corners", y_corners)

        # self.conv3d = nn.Conv2d((len(y_corners)-1) * channels, channels,1)
        self.conv3d = nn.Linear((len(y_corners) - 1) * channels, channels)
        self.scale = scale

    def forward(self, features, calib, grid):

        # Expand the grid in the y dimension
        corners = grid.unsqueeze(1) + self.y_corners.view(-1, 1, 1, 3)

        # Project grid corners to image plane and normalize to [-1, 1]
        img_corners = utils.perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        # Normalize to [-1, 1]
        img_height, img_width = features.size()[2:]
        img_size = corners.new([img_width, img_height]) / self.scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat(
            [
                torch.min(
                    norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]
                ),
                torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
            ],
            dim=-1,
        )
        batch, _, depth, width, _ = bbox_corners.size()
        bbox_corners = bbox_corners.flatten(2, 3)

        # Compute the area of each bounding box
        area = (
            (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1)
            * img_height
            * img_width
            * 0.25
            + EPSILON
        ).unsqueeze(1)
        visible = area > EPSILON

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]])
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]])
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]])
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]])

        # Compute voxel features (ignore features which are not visible)
        vox_feats = (top_left + btm_right - top_right - btm_left) / area
        vox_feats = vox_feats * visible.float()
        # vox_feats = vox_feats.view(batch, -1, depth, width)
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)

        # Flatten to orthographic feature map
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)
        # ortho_feats = F.relu(self.conv3d(vox_feats))

        # Block gradients to pixels which are not visible in the image

        return ortho_feats


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
