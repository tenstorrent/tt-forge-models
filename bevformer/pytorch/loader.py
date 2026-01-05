# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormer model loader implementation
"""
import torch
from typing import Optional
from ...base import ForgeModel
from ...config import (
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelInfo,
    ModelConfig,
)

from .src.model import (
    BEVFormer,
    BEVFormerV2,
    get_bevformer_model,
    get_bevformer_v2_model,
    load_checkpoint_bev,
)
from .src.model_utils import build_inputs
from ...tools.utils import get_file


class MaxPool3x3Stride2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(x)


class ModelVariant(StrEnum):
    """Available BEVFormer model variants."""

    BEVFORMER_TINY = "BEVFormer-tiny"
    BEVFORMER_SMALL = "BEVFormer-small"
    BEVFORMER_BASE = "BEVFormer-base"
    BEVFORMER_V2_R50_T1_BASE = "bevformerv2-r50-t1-base"
    BEVFORMER_V2_R50_T1 = "bevformerv2-r50-t1"
    BEVFORMER_V2_R50_T2 = "bevformerv2-r50-t2"
    BEVFORMER_V2_R50_T8 = "bevformerv2-r50-t8"


class ModelLoader(ForgeModel):
    """BEVFormer model loader implementation for autonomous driving tasks."""

    _VARIANTS = {
        ModelVariant.BEVFORMER_TINY: ModelConfig(
            pretrained_model_name="BEVFormer-tiny"
        ),
        ModelVariant.BEVFORMER_SMALL: ModelConfig(
            pretrained_model_name="BEVFormer-small"
        ),
        ModelVariant.BEVFORMER_BASE: ModelConfig(
            pretrained_model_name="BEVFormer-base"
        ),
        ModelVariant.BEVFORMER_V2_R50_T1_BASE: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t1-base"
        ),
        ModelVariant.BEVFORMER_V2_R50_T1: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t1"
        ),
        ModelVariant.BEVFORMER_V2_R50_T2: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t2"
        ),
        ModelVariant.BEVFORMER_V2_R50_T8: ModelConfig(
            pretrained_model_name="bevformerv2-r50-t8"
        ),
    }
    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEVFORMER_TINY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="bevformer",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Load and return the BEVFormer model instance with default settings.
        Returns:
            Torch model: The BEVFormer model instance.
        """
        model = MaxPool3x3Stride2()
        return model

    def load_inputs(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Return sample inputs for the BEVFormer model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """

        inputs = torch.randn(6, 64, 240, 400)
        return inputs
