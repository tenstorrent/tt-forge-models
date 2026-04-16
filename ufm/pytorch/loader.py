# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UFM (UniFlowMatch) model loader implementation for dense correspondence estimation
"""

import torch
import torch.nn as nn
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


class UFMWrapper(nn.Module):
    """Wrapper around UFM model that accepts simple tensor inputs.

    The original UFM forward() takes two view dicts with non-tensor metadata.
    This wrapper accepts BCHW image tensors directly and constructs the
    required view dicts internally.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.data_norm_type = model.encoder.data_norm_type

    def forward(self, img1, img2):
        view1 = {
            "img": img1,
            "symmetrized": False,
            "data_norm_type": self.data_norm_type,
        }
        view2 = {
            "img": img2,
            "symmetrized": False,
            "data_norm_type": self.data_norm_type,
        }
        return self.model(view1, view2)


class ModelVariant(StrEnum):
    """Available UFM model variants."""

    REFINE = "Refine"


class ModelLoader(ForgeModel):
    """UFM model loader implementation for dense correspondence estimation."""

    _VARIANTS = {
        ModelVariant.REFINE: ModelConfig(
            pretrained_model_name="infinity1096/UFM-Refine",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REFINE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        from uniflowmatch import UniFlowMatchClassificationRefinement

        model = UniFlowMatchClassificationRefinement.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return UFMWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        # UFM forward expects two BCHW image tensors normalized for DINOv2
        height, width = 420, 560
        dtype = dtype_override or torch.float32
        img1 = torch.randn(batch_size, 3, height, width, dtype=dtype)
        img2 = torch.randn(batch_size, 3, height, width, dtype=dtype)

        return {"img1": img1, "img2": img2}
