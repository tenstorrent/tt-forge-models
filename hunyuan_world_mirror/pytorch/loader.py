# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanWorld-Mirror model loader implementation for 3D geometric prediction.

Loads the WorldMirror model, a feed-forward architecture for universal 3D world
reconstruction from images. It performs camera pose estimation, depth prediction,
point cloud generation, surface normal estimation, and 3D Gaussian generation.

Requires the HunyuanWorld-Mirror repository to be cloned at /tmp/hunyuan_world_mirror_repo.
"""
import os
import sys

import torch
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

REPO_PATH = "/tmp/hunyuan_world_mirror_repo"


def _ensure_repo_importable():
    """Ensure the HunyuanWorld-Mirror repo is cloned and importable."""
    if REPO_PATH not in sys.path:
        if not os.path.isdir(REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git",
                    REPO_PATH,
                ]
            )

        sys.path.insert(0, REPO_PATH)


def _patch_geometry():
    """Patch upstream geometry utilities to preserve dtype instead of forcing float32.

    Two bugs:
    1. depth_to_camera_coords calls camera_intrinsics.float() causing bfloat16/float32
       mismatch in the einsum with R_cam_to_world (model output in bfloat16).
    2. GaussianSplatRenderer.prune_gs uses torch.zeros(...) without dtype, defaulting
       to float32, while splat weights from the model are bfloat16.
    """
    import src.models.utils.geometry as geometry

    def _fixed_depth_to_camera_coords(depthmap, camera_intrinsics):
        B, H, W = depthmap.shape
        device = depthmap.device
        dtype = depthmap.dtype

        camera_intrinsics = camera_intrinsics.to(dtype)

        fx = camera_intrinsics[:, 0, 0]
        fy = camera_intrinsics[:, 1, 1]
        cx = camera_intrinsics[:, 0, 2]
        cy = camera_intrinsics[:, 1, 2]

        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device),
            torch.arange(W, dtype=dtype, device=device),
            indexing="ij",
        )
        u_grid = u_grid.unsqueeze(0)
        v_grid = v_grid.unsqueeze(0)

        z_cam = depthmap
        x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
        y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)

        X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        valid_mask = depthmap > 0.0

        return X_cam, valid_mask

    geometry.depth_to_camera_coords = _fixed_depth_to_camera_coords

    import src.models.models.rasterization as rasterization

    _orig_prune_gs = rasterization.GaussianSplatRenderer.prune_gs

    def _fixed_prune_gs(self, splats, voxel_size=0.002):
        # Determine dtype from input splats so torch.zeros uses the correct dtype.
        dtype = splats["means"].dtype
        B = splats["means"].shape[0]
        merged_splats_list = []
        device = splats["means"].device

        for i in range(B):
            splats_i = {k: splats[k][i] for k in ["means", "quats", "scales", "opacities", "sh", "weights"]}

            coords = splats_i["means"]
            voxel_indices = (coords / voxel_size).floor().long()
            min_indices = voxel_indices.min(dim=0)[0]
            voxel_indices = voxel_indices - min_indices
            max_dims = voxel_indices.max(dim=0)[0] + 1

            flat_indices = (
                voxel_indices[:, 0] * max_dims[1] * max_dims[2]
                + voxel_indices[:, 1] * max_dims[2]
                + voxel_indices[:, 2]
            )

            unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            K = len(unique_voxels)

            merged = {
                "means": torch.zeros((K, 3), device=device, dtype=dtype),
                "quats": torch.zeros((K, 4), device=device, dtype=dtype),
                "scales": torch.zeros((K, 3), device=device, dtype=dtype),
                "opacities": torch.zeros(K, device=device, dtype=dtype),
                "sh": torch.zeros((K, self.nums_sh, 3), device=device, dtype=dtype),
            }

            weights = splats_i["weights"]
            weight_sums = torch.zeros(K, device=device, dtype=dtype)
            weight_sums.scatter_add_(0, inverse_indices, weights)
            weight_sums = torch.clamp(weight_sums, min=1e-8)

            for d in range(3):
                merged["means"][:, d].scatter_add_(0, inverse_indices, splats_i["means"][:, d] * weights)
            merged["means"] = merged["means"] / weight_sums.unsqueeze(1)

            for d in range(3):
                merged["sh"][:, 0, d].scatter_add_(0, inverse_indices, splats_i["sh"][:, 0, d] * weights)
            merged["sh"] = merged["sh"] / weight_sums.unsqueeze(-1).unsqueeze(-1)

            merged["opacities"].scatter_add_(0, inverse_indices, weights * weights)
            merged["opacities"] = merged["opacities"] / weight_sums

            for d in range(3):
                merged["scales"][:, d].scatter_add_(0, inverse_indices, splats_i["scales"][:, d] * weights)
            merged["scales"] = merged["scales"] / weight_sums.unsqueeze(1)

            for d in range(4):
                merged["quats"][:, d].scatter_add_(0, inverse_indices, splats_i["quats"][:, d] * weights)
            quat_norms = torch.norm(merged["quats"], dim=1, keepdim=True)
            merged["quats"] = merged["quats"] / torch.clamp(quat_norms, min=1e-8)

            merged_splats_list.append(merged)

        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats"]:
            output[key] = [merged[key] for merged in merged_splats_list]

        return output

    rasterization.GaussianSplatRenderer.prune_gs = _fixed_prune_gs


class ModelVariant(StrEnum):
    """Available HunyuanWorld-Mirror model variants."""

    WORLD_MIRROR = "World_Mirror"


class ModelLoader(ForgeModel):
    """HunyuanWorld-Mirror model loader for universal 3D geometric prediction."""

    _VARIANTS = {
        ModelVariant.WORLD_MIRROR: ModelConfig(
            pretrained_model_name="tencent/HunyuanWorld-Mirror",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WORLD_MIRROR

    # Model architecture constants
    _IMG_SIZE = 518
    _NUM_VIEWS = 2  # Minimum number of input views for 3D reconstruction

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HunyuanWorld-Mirror",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WorldMirror model.

        Returns:
            torch.nn.Module: The WorldMirror 3D geometric prediction model.
        """
        _ensure_repo_importable()
        from src.models.models.worldmirror import WorldMirror

        _patch_geometry()

        repo_id = self._variant_config.pretrained_model_name
        model = WorldMirror.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the WorldMirror model.

        Returns:
            dict: Input dict with 'views' and 'cond_flags' for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # img: input images [B, N, 3, H, W] in [0, 1]
        img = torch.rand(
            batch_size,
            self._NUM_VIEWS,
            3,
            self._IMG_SIZE,
            self._IMG_SIZE,
            dtype=dtype,
        )

        views = {"img": img}
        cond_flags = [0, 0, 0]  # No optional priors: [camera_pose, depth, intrinsics]

        return {"views": views, "cond_flags": cond_flags}
