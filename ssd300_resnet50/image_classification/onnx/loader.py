# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD300 ResNet50 image classification ONNX model loader implementation
"""
import onnx

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file
from .src.model_utils import get_input_img


class ModelLoader(ForgeModel):
    """SSD300 ResNet50 image classification ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with specified variant."""
        super().__init__()
        self.variant = "ssd300_resnet50"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="ssd300_resnet50",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the SSD300 ResNet50 ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/ssd300_resnet50.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch SSD300 ResNet50 preprocessing."""
        inputs = get_input_img()
        return [inputs]
