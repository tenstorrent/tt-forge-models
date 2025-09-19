# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function


class VoxelPoolingInference(Function):
    @staticmethod
    def forward(
        ctx,
        geom_xyz: torch.Tensor,
        depth_features: torch.Tensor,
        context_features: torch.Tensor,
        voxel_num: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function for `voxel pooling.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, N, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3].

        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert depth_features.is_contiguous()
        assert context_features.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        batch_size = geom_xyz.shape[0]
        num_cams = geom_xyz.shape[1]
        num_depth = geom_xyz.shape[2]
        num_height = geom_xyz.shape[3]
        num_width = geom_xyz.shape[4]
        num_channels = context_features.shape[1]
        # If tensors are on CUDA, use the optimized extension. Otherwise run a CPU fallback.
        if geom_xyz.is_cuda and depth_features.is_cuda and context_features.is_cuda:
            output_features = depth_features.new_zeros(
                (batch_size, voxel_num[1], voxel_num[0], num_channels)
            )
            voxel_pooling_inference_ext.voxel_pooling_inference_forward_wrapper(
                batch_size,
                num_cams,
                num_depth,
                num_height,
                num_width,
                num_channels,
                voxel_num[0],
                voxel_num[1],
                voxel_num[2],
                geom_xyz,
                depth_features,
                context_features,
                output_features,
            )
            return output_features.permute(0, 3, 1, 2)

        # CPU fallback implementation
        # Shapes:
        # - geom_xyz: [B, Cams, D, H, W, 3] (int)
        # - depth_features: [B*Cams, D, H, W]
        # - context_features: [B*Cams, Cfeat, H, W]
        # Output: [B, Cfeat, Vy, Vx]
        assert (
            not geom_xyz.is_cuda
            and not depth_features.is_cuda
            and not context_features.is_cuda
        ), "CPU fallback expects all inputs on CPU"

        # Ensure dtypes
        if geom_xyz.dtype != torch.int32 and geom_xyz.dtype != torch.int64:
            raise RuntimeError("geom_xyz must be integer tensor on CPU path")

        # Allocate output on CPU with same dtype as context features
        voxel_num_x = int(voxel_num[0].detach().cpu().item())
        voxel_num_y = int(voxel_num[1].detach().cpu().item())
        voxel_num_z = int(voxel_num[2].detach().cpu().item())
        output_features = context_features.new_zeros(
            (batch_size, voxel_num_y, voxel_num_x, num_channels)
        )

        # Make local views for speed
        B = batch_size
        Cams = num_cams
        D = num_depth
        H = num_height
        W = num_width
        Vy = voxel_num_y
        Vx = voxel_num_x
        Vz = voxel_num_z

        # Vectorized accumulation using scatter-add on flattened output
        # Flatten spatial samples across (B, Cams, D, H, W)
        total_samples = B * Cams * D * H * W
        # [N, 3] voxel coordinates
        geom_flat = geom_xyz.view(total_samples, 3)
        sx_all = geom_flat[:, 0]
        sy_all = geom_flat[:, 1]
        sz_all = geom_flat[:, 2]

        # Valid mask inside voxel bounds
        valid = (
            (sx_all >= 0)
            & (sx_all < Vx)
            & (sy_all >= 0)
            & (sy_all < Vy)
            & (sz_all >= 0)
            & (sz_all < Vz)
        )
        if valid.any():
            sx_v = sx_all[valid].to(torch.long)
            sy_v = sy_all[valid].to(torch.long)
            # derive indices (b, cam, d, h, w) from flat indices
            idx = torch.arange(total_samples, device=geom_xyz.device)[valid]
            # compute divisions on CPU tensors
            denom_cdhw = Cams * D * H * W
            denom_dhw = D * H * W
            denom_hw = H * W
            b_idx = torch.div(idx, denom_cdhw, rounding_mode="floor")
            cam_idx = torch.div(idx, denom_dhw, rounding_mode="floor") % Cams
            d_idx = torch.div(idx, denom_hw, rounding_mode="floor") % D
            h_idx = torch.div(idx, W, rounding_mode="floor") % H
            w_idx = idx % W
            bc_idx = b_idx * Cams + cam_idx

            # Gather depth [N]
            depth_vals = depth_features[bc_idx, d_idx, h_idx, w_idx]
            nonzero = depth_vals != 0
            if nonzero.any():
                # Filter by non-zero depth
                depth_vals = depth_vals[nonzero]
                bc_nz = bc_idx[nonzero]
                b_nz = b_idx[nonzero]
                sx_nz = sx_v[nonzero]
                sy_nz = sy_v[nonzero]
                h_nz = h_idx[nonzero]
                w_nz = w_idx[nonzero]
                # Gather context vectors [N, C]
                ctx_vecs = context_features[bc_nz, :, h_nz, w_nz]
                contrib = ctx_vecs * depth_vals.unsqueeze(1)
                # Flatten output over (B, Vy, Vx) -> [B*Vy*Vx, C]
                out_flat = output_features.view(B * Vy * Vx, num_channels)
                out_lin_idx = b_nz * (Vy * Vx) + sy_nz * Vx + sx_nz
                out_flat.index_add_(0, out_lin_idx, contrib)

        return output_features.permute(0, 3, 1, 2)


voxel_pooling_inference = VoxelPoolingInference.apply
