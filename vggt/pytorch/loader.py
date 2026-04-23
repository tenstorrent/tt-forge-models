# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGGT (Visual Geometry Grounded Transformer) model loader for image-to-3D
scene inference (camera pose, depth, point maps).
"""

import os

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
from .src import VGGT


class ModelVariant(StrEnum):
    """Available VGGT model variants."""

    VGGT_1B_COMMERCIAL = "1B-Commercial"


class ModelLoader(ForgeModel):
    """VGGT model loader for 3D scene attribute inference from images."""

    _VARIANTS = {
        ModelVariant.VGGT_1B_COMMERCIAL: ModelConfig(
            pretrained_model_name="facebook/VGGT-1B-Commercial",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VGGT_1B_COMMERCIAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VGGT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = VGGT()
        else:
            try:
                model = VGGT.from_pretrained(pretrained_model_name)
            except Exception:
                model = VGGT()

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.float32

        # Multi-view image input: [B, S, 3, H, W] - 2 views of 518x518 images
        images = torch.randn(batch_size, 2, 3, 518, 518, dtype=dtype)
        # Clamp to [0, 1] range as expected by the model
        images = images.clamp(0, 1)

        return {"images": images}
