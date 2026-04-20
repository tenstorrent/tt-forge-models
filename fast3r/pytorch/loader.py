# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fast3R model loader implementation for multi-view 3D reconstruction.

Fast3R processes many images in a single forward pass to produce per-view
point maps and camera poses, following the DUSt3R-style multi-view
reconstruction formulation.

Requires the fast3r repository to be cloned at /tmp/fast3r_repo.
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

FAST3R_REPO_PATH = "/tmp/fast3r_repo"


def _ensure_fast3r_importable():
    """Ensure the fast3r repo is cloned, installed, and importable."""
    if not os.path.isdir(FAST3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/facebookresearch/fast3r.git",
                FAST3R_REPO_PATH,
            ]
        )
        subprocess.check_call(
            ["pip", "install", "-e", FAST3R_REPO_PATH],
        )

    if FAST3R_REPO_PATH not in sys.path:
        sys.path.insert(0, FAST3R_REPO_PATH)


class ModelVariant(StrEnum):
    """Available Fast3R model variants."""

    VIT_LARGE_512 = "ViT_Large_512"


class ModelLoader(ForgeModel):
    """Fast3R model loader for multi-view 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.VIT_LARGE_512: ModelConfig(
            pretrained_model_name="jedyang97/Fast3R_ViT_Large_512",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_LARGE_512

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Fast3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fast3R model.

        Returns:
            torch.nn.Module: The Fast3R multi-view reconstruction model.
        """
        _ensure_fast3r_importable()
        from fast3r.models.fast3r import Fast3R

        repo_id = self._variant_config.pretrained_model_name
        model = Fast3R.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample multi-view inputs for Fast3R.

        Fast3R's forward method expects a list of view dicts, each containing
        an 'img' tensor and a 'true_shape' metadata tensor.

        Returns:
            list[dict]: List of view dicts for model(views=...) invocation.
        """
        dtype = dtype_override or torch.float32
        height, width = 512, 512
        num_views = 2

        torch.manual_seed(42)

        views = [
            {
                "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
                "true_shape": torch.tensor([[height, width]]).repeat(batch_size, 1),
            }
            for _ in range(num_views)
        ]

        return views
