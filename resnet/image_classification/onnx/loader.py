# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet image classification ONNX model loader implementation
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
from transformers import ResNetForImageClassification, AutoImageProcessor
from datasets import load_dataset


class ModelLoader(ForgeModel):
    """ResNet image classification ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with specified variant."""
        super().__init__()
        self.variant = "resnet50"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="resnet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNet ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/resnet50.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        self.pt_model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        )
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch EfficientNet preprocessing."""
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        inputs = processor(image, return_tensors="pt")
        input_sample = inputs["pixel_values"]
        return [input_sample]

    def print_cls_results(self, outputs):
        """Decode logits to top-k class labels and probabilities.

        Args:
            outputs: Model outputs (tensor or list/tuple with logits at index 0).

        Returns:
            str: The top 1 predicted class label.
        """
        predicted_label = outputs[0].argmax(-1).item()
        return self.pt_model.config.id2label[predicted_label]
