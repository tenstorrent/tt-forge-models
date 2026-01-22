#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 object detection ONNX model loader implementation
"""
import onnx
from torchvision import transforms
from datasets import load_dataset

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel
from ....tools.utils import get_file


class ModelLoader(ForgeModel):
    """YOLOv8 object detection ONNX model loader implementation."""

    def __init__(self):
        """Initialize ModelLoader with default variant."""
        super().__init__()
        self.variant = "yolov8"

    def _get_model_info(self, variant: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="yolov8",
            variant=self.variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.ONNX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv8 ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        path = "/proj_sw/user_dev/mramanathan/bgdlab22_jan7_forge/tt-forge-fe/onnx_dir/yolov8.onnx"
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for YOLOv8 ONNX model."""
        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image).unsqueeze(0)

        return [image_tensor]
