# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet image classification ONNX model loader implementation
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
import timm
from datasets import load_dataset


class ModelLoader(ForgeModel):
    """EfficientNet image classification ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with specified variant."""
        super().__init__()
        self.variant = "efficientnet_b0"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="efficientnet",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the EfficientNet ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = f"/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/efficientnet_b0_timm.onnx"
        # file = get_file(path)

        # Load and validate ONNX model
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs using the PyTorch EfficientNet preprocessing."""
        model = timm.create_model(self.variant, pretrained=True)
        dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        img = next(iter(dataset.skip(10)))["image"]
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        img_tensor = transforms(img).unsqueeze(0)
        return [img_tensor]

    def decode_output(self, outputs):
        """Decode logits to top-k class labels and probabilities.

        Args:
            outputs: Model outputs (tensor or list/tuple with logits at index 0).

        Returns:
            str: The top 1 predicted class label.
        """
        print_compiled_model_results(outputs)
