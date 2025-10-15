# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv10 model loader implementation
"""
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
from ...tools.utils import yolo_postprocess
from loguru import logger


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ip):
        x = ip.topk(300)[1]
        return x


class ModelVariant(StrEnum):
    """Available YOLOv10 model variants."""

    YOLOV10X = "yolov10x"
    YOLOV10N = "yolov10n"


class ModelLoader(ForgeModel):
    """YOLOv10 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV10X: ModelConfig(
            pretrained_model_name="yolov10x",
        ),
        ModelVariant.YOLOV10N: ModelConfig(
            pretrained_model_name="yolov10n",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV10X

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [ModelVariant.YOLOV10X]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="yolov10",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv10 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv10 model instance.
        """

        model = Wrapper()
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv10 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        torch.set_printoptions(edgeitems=100, threshold=10000, linewidth=200)

        variant = self._variant_config.pretrained_model_name

        if variant == "yolov10n":
            s = torch.load("yolov10n_scores.pt")
        elif variant == "yolov10x":
            s = torch.load("yolov10x_scores.pt")

        logger.info("s={}", s)
        logger.info("s.dtype={}", s.dtype)
        logger.info("s.shape={}", s.shape)

        return s

    def post_process(self, co_out):
        """Post-process YOLOv10 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv10 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)
