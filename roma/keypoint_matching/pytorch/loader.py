# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoMa model loader implementation for dense feature / keypoint matching.

The vismatch/roma Hugging Face repo is a metadata marker for the vismatch
library; the underlying model and pretrained weights are provided by the
`romatch` package and downloaded via torch.hub at load time.
"""

from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available RoMa model variants for keypoint matching."""

    ROMA_OUTDOOR = "roma_outdoor"


class ModelLoader(ForgeModel):
    """RoMa model loader implementation for keypoint matching tasks."""

    _VARIANTS = {
        ModelVariant.ROMA_OUTDOOR: ModelConfig(
            pretrained_model_name="vismatch/roma",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROMA_OUTDOOR

    # Default coarse matching resolution used by roma_outdoor.
    coarse_res = 560

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RoMa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_KEYPOINT_MATCHING,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from romatch import roma_outdoor

        # roma_outdoor hard-asserts the highest matmul precision.
        torch.set_float32_matmul_precision("highest")

        model = roma_outdoor(device="cpu", **kwargs)
        model.eval()
        model = model.to(dtype_override or torch.float32)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.float32
        res = self.coarse_res
        batch = {
            "im_A": torch.randn(batch_size, 3, res, res, dtype=dtype),
            "im_B": torch.randn(batch_size, 3, res, res, dtype=dtype),
        }
        return {"batch": batch}
