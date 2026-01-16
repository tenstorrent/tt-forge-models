# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG image classification ONNX model loader implementation
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
from .src.model_utils import get_input_img_osmr
from ....tools.utils import print_compiled_model_results


class ModelLoader(ForgeModel):
    """VGG image classification ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with specified variant."""
        super().__init__()
        self.variant = "vgg11"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="vgg",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the VGG ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/vgg11.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch OSMR preprocessing."""
        inputs = get_input_img_osmr()
        return [inputs]

    def print_cls_results(self, outputs):
        """Print the classification results."""
        print_compiled_model_results(outputs)
