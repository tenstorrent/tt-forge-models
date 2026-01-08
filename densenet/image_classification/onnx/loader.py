# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DenseNet image classification ONNX model loader implementation
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
from .src.model_utils import get_input_img_torchvision


class ModelLoader(ForgeModel):
    """DenseNet image classification ONNX model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. Defaults to "densenet121".

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="densenet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the DenseNet ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        # Download the model
        model_path = hf_hub_download(
            repo_id="meenakshiramanathan1/onnx", filename="densenet121.onnx"
        )

        # Load and validate ONNX model
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for DenseNet image classification.

        Returns:
            torch.Tensor: Input tensor
        """
        inputs = get_input_img_torchvision()
        return [inputs]

    def decode_output(self, outputs):
        """Decode logits to top-k class labels and probabilities.

        Args:
            outputs: Model outputs (tensor or list/tuple with logits at index 0).

        Returns:
            str: The top 1 predicted class label.
        """
        print_compiled_model_results(outputs)
