# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MASt3R (Matching And Stereo 3D Reconstruction) model loader implementation.

Requires the mast3r repository to be cloned at /tmp/mast3r_repo.
"""
import os
import sys

import torch
from typing import Optional

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

MAST3R_REPO_PATH = "/tmp/mast3r_repo"


def _ensure_mast3r_importable():
    """Ensure the mast3r repo is cloned and importable."""
    if not os.path.isdir(MAST3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/naver/mast3r.git",
                MAST3R_REPO_PATH,
            ]
        )

    if MAST3R_REPO_PATH not in sys.path:
        sys.path.insert(0, MAST3R_REPO_PATH)
    dust3r_path = os.path.join(MAST3R_REPO_PATH, "dust3r")
    if dust3r_path not in sys.path:
        sys.path.insert(0, dust3r_path)


class ModelVariant(StrEnum):
    """Available MASt3R model variants."""

    VIT_LARGE_BASE_DECODER_512 = "ViTLarge_BaseDecoder_512"


class ModelLoader(ForgeModel):
    """MASt3R stereo 3D reconstruction model loader implementation."""

    _VARIANTS = {
        ModelVariant.VIT_LARGE_BASE_DECODER_512: ModelConfig(
            pretrained_model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_LARGE_BASE_DECODER_512

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MASt3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MASt3R model instance."""
        _ensure_mast3r_importable()
        from mast3r.model import AsymmetricMASt3R

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AsymmetricMASt3R.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample stereo image pair inputs for the MASt3R model.

        Returns a list of two view dicts, each containing an image tensor
        and its true shape, matching the model's expected input format.
        """
        dtype = dtype_override or torch.float32
        height, width = 384, 512

        torch.manual_seed(42)

        view1 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]]).repeat(batch_size, 1),
        }
        view2 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]]).repeat(batch_size, 1),
        }

        return [view1, view2]
