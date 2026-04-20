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
import types
from unittest.mock import MagicMock

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


def _patch_cuda_autocast():
    """Replace torch.amp.autocast 'cuda' calls with 'cpu' for non-CUDA environments."""
    _original_autocast = torch.amp.autocast

    class _PatchedAutocast(_original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            if device_type == "cuda" and not torch.cuda.is_available():
                device_type = "cpu"
                if "dtype" in kwargs and kwargs["dtype"] == torch.float32:
                    kwargs["enabled"] = False
            super().__init__(device_type, *args, **kwargs)

    torch.amp.autocast = _PatchedAutocast


def _patch_linalg_for_bfloat16(model):
    """Patch transform_camera_vector to cast to float32 for linalg.inv."""
    original_fn = model.transform_camera_vector

    def patched_transform_camera_vector(camera_params, h, w):
        orig_dtype = camera_params.dtype
        if orig_dtype == torch.bfloat16:
            camera_params = camera_params.float()
        c2w_mat, int_mat = original_fn(camera_params, h, w)
        return c2w_mat.to(orig_dtype), int_mat.to(orig_dtype)

    model.transform_camera_vector = patched_transform_camera_vector


def _mock_gsplat():
    """Mock gsplat module which requires CUDA to install."""
    if "gsplat" not in sys.modules:
        gsplat = types.ModuleType("gsplat")
        gsplat.rendering = MagicMock()
        gsplat.strategy = MagicMock()
        sys.modules["gsplat"] = gsplat
        sys.modules["gsplat.rendering"] = gsplat.rendering
        sys.modules["gsplat.strategy"] = gsplat.strategy


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
        _patch_cuda_autocast()
        _mock_gsplat()
        _ensure_repo_importable()
        from src.models.models.worldmirror import WorldMirror

        repo_id = self._variant_config.pretrained_model_name
        model = WorldMirror.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        _patch_linalg_for_bfloat16(model)

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
