# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv11n model loader implementation
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
from .src.yolov11n import YoloV11n


class ModelVariant(StrEnum):
    """Available YOLOv11n model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """YOLOv11n model loader implementation.

    Uses randomly initialized weights. No external dependencies (ultralytics) required.
    """

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="YOLOv11n",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return a YOLOv11n model instance with random weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The YOLOv11n model instance.
        """
        model = YoloV11n()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return a random sample input tensor.

        YOLOv11n expects input of shape [batch, 3, 640, 640].

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            torch.Tensor: Random input tensor of shape [batch_size, 3, 640, 640].
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(batch_size, 3, 640, 640, dtype=dtype)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output for backward pass.

        YOLOv11n returns a single detection tensor of shape [batch, 84, N]
        where N is the total number of anchor points across all scales.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Flattened output tensor for backward pass.
        """
        if isinstance(fwd_output, (tuple, list)):
            flattened = [t.flatten(start_dim=1) for t in fwd_output]
            return torch.cat(flattened, dim=1)
        return fwd_output
