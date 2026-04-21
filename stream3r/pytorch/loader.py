# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
STream3R model loader implementation for streaming 3D reconstruction.

STream3R reformulates 3D reconstruction as a decoder-only Transformer problem,
using causal attention with optional KV cache and sliding window attention to
produce pointmap predictions from sequences of images.

Requires the STream3R repository to be cloned at /tmp/stream3r_repo.
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

STREAM3R_REPO_PATH = "/tmp/stream3r_repo"


def _ensure_stream3r_importable():
    """Ensure the STream3R repo is cloned and importable."""
    if not os.path.isdir(STREAM3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/NIRVANALAN/STream3R.git",
                STREAM3R_REPO_PATH,
            ]
        )

    if STREAM3R_REPO_PATH not in sys.path:
        sys.path.insert(0, STREAM3R_REPO_PATH)


class ModelVariant(StrEnum):
    """Available STream3R model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """STream3R model loader for streaming 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="yslan/STream3R",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="STream3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the STream3R model.

        Returns:
            torch.nn.Module: The STream3R streaming 3D reconstruction model.
        """
        _ensure_stream3r_importable()
        from stream3r.models.stream3r import STream3R

        repo_id = self._variant_config.pretrained_model_name
        model = STream3R.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample image sequence inputs for STream3R.

        STream3R expects a tensor of shape [S, 3, H, W] or [B, S, 3, H, W]
        with pixel values normalized to the [0, 1] range. Spatial dimensions
        must be divisible by the patch size (14); the reference resolution is
        close to 518x384 so we use the nearest multiples (518 = 37*14, 378 = 27*14).

        Returns:
            dict: Dict with 'images' and 'mode' keys for model(**inputs) unpacking.
        """
        dtype = dtype_override or torch.float32
        num_frames, height, width = 4, 378, 518

        torch.manual_seed(42)

        images = torch.rand(batch_size, num_frames, 3, height, width, dtype=dtype)

        return {"images": images, "mode": "causal"}
