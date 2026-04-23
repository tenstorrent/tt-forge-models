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

    def load_model(self, **kwargs):
        from romatch import roma_outdoor

        # roma_outdoor hard-asserts the highest matmul precision.
        torch.set_float32_matmul_precision("highest")

        # use_custom_corr=False uses the pure-Python local correlation fallback
        # to avoid depending on the 'local_corr' C extension that is not bundled
        # in the romatch PyPI wheel.
        model = roma_outdoor(device="cpu", use_custom_corr=False)
        model.eval()

        return model

    def load_inputs(self, batch_size=1):
        res = self.coarse_res
        batch = {
            "im_A": torch.randn(batch_size, 3, res, res),
            "im_B": torch.randn(batch_size, 3, res, res),
        }
        return {"batch": batch}
