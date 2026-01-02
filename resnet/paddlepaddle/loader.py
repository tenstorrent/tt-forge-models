# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet PaddlePaddle model loader implementation.
"""

from typing import Optional

import paddle

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


class ModelVariant(StrEnum):
    """Available ResNet model variants (Paddle)."""

    DEFAULT = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"


class ModelLoader(ForgeModel):
    """ResNet PaddlePaddle model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="resnet18",
        ),
        ModelVariant.RESNET34: ModelConfig(
            pretrained_model_name="resnet34",
        ),
        ModelVariant.RESNET50: ModelConfig(
            pretrained_model_name="resnet50",
        ),
        ModelVariant.RESNET101: ModelConfig(
            pretrained_model_name="resnet101",
        ),
        ModelVariant.RESNET152: ModelConfig(
            pretrained_model_name="resnet152",
        ),
    }

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="resnet",
            variant=variant.value,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.PADDLE,
        )

    def load_model(self, variant: ModelVariant, dtype_override=None):
        """Load pretrained ResNet model for the given variant (Paddle)."""
        model = eval(variant.value)(pretrained=True)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for ResNet model (Paddle)."""
        inputs = paddle.rand([batch_size, 3, 224, 224])
        return [inputs]
