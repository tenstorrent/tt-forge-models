# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CenterPoint model loader — https://github.com/tianweiy/CenterPoint (CVPR 2021).

Uses the det3d submodule from Toyota-fresh (same code path as Toyota PR #5)
so that we exercise the exact same model graph as the Toyota bringup test.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

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

# Path to Toyota-fresh centerpoint utils (contains load_model_with_weights)
_TOYOTA_CENTERPOINT = (
    Path(__file__).resolve().parents[4] / "Toyota-fresh" / "centerpoint"
)

def _get_toyota_utils():
    """Import load_model_with_weights from Toyota-fresh centerpoint utils."""
    toyota_root = str(_TOYOTA_CENTERPOINT.parent)
    if toyota_root not in sys.path:
        sys.path.insert(0, toyota_root)
    from centerpoint.model.utils import load_model_with_weights, get_single_input
    return load_model_with_weights, get_single_input



@dataclass
class CenterPointConfig(ModelConfig):
    pass


class ModelVariant(StrEnum):
    CENTERPOINT_PILLAR = "CenterPoint_Pillar"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CENTERPOINT_PILLAR: CenterPointConfig(
            pretrained_model_name="centerpoint_pillar",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CENTERPOINT_PILLAR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CenterPoint",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MULTIVIEW_3D_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        load_model_with_weights, _ = _get_toyota_utils()
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = load_model_with_weights(dtype=dtype)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        _, get_single_input = _get_toyota_utils()
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        bev = get_single_input(dtype=dtype, batch_size=batch_size)
        return (bev,)
