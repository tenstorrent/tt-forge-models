# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GoogLeNet image classification ONNX model loader implementation
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
from ....tools.utils import print_compiled_model_results
from ...pytorch import ModelLoader as PTModelLoader


class ModelLoader(ForgeModel):
    """GoogLeNet image classification ONNX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which torchvision GoogLeNet variant to use.
                     Examples: "googlenet".
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="googlenet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the GoogLeNet ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/googlenet.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch GoogLeNet preprocessing."""
        pt_loader = PTModelLoader()
        inputs = pt_loader.load_inputs()
        inputs = inputs.contiguous()
        return [inputs]

    def print_cls_results(self, outputs):
        """Decode logits to top-k class labels and probabilities.

        Args:
            outputs: Model outputs (tensor or list/tuple with logits at index 0).

        Returns:
            str: The top 1 predicted class label.
        """
        print_compiled_model_results(outputs)
