# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobileNetV2 model loader implementation
"""
import torch
import onnx
from PIL import Image
from torchvision import transforms
import numpy as np

from ...base import ModelLoader
from ...tools.utils import get_file


class OnnxMobilenetv2ModelLoader(ModelLoader):
    """MobileNetV2 model loader implementation."""

    @classmethod
    def load_model(cls):
        """Load and return the MobileNetV2 model instance with default settings.

        Returns:
            Onnx model: The MobileNetV2 model instance.
        """
        # Load model with defaults
        file = get_file("test_files/onnx/mobilenetv2_100/mobilenetv2_100.onnx")
        model = onnx.load(file)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the MobileNetV2 model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        # Original image used in test
        image_file = get_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
        )

        # Download and load image
        image = Image.open(image_file)

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        img_tensor = [transform(image).unsqueeze(0)]
        batch_tensor = torch.cat(img_tensor, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
