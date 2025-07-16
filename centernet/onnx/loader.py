# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterNet model loader implementation
"""
import torch
import onnx
from PIL import Image
from torchvision import transforms
import numpy as np
from enum import StrEnum
from typing import Optional

from ...config import (
    ModelInfo,
    ModelConfig,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """CenterNet model loader implementation."""
    
    class Variant(StrEnum):
        DLA1X_OD = "dla1x_od"  # Default variant for object detection
        DLA1X_HPE = "dla1x_hpe"  # Human pose estimation
        DLA1X_3D = "dla1x_3d"  # 3D detection
    
    _VARIANTS = {
        Variant.DLA1X_OD: ModelConfig(
            pretrained_model_name="",
        ),
        Variant.DLA1X_HPE: ModelConfig(
            pretrained_model_name="",
        ),
        Variant.DLA1X_3D: ModelConfig(
            pretrained_model_name="",
        ),
    }
    
    DEFAULT_VARIANT = Variant.DLA1X_OD
    
    def __init__(self, variant: Optional[StrEnum] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        # Determine task based on variant name
        if variant is not None and "hpe" in str(variant).lower():
            task = ModelTask.CV_POSE_ESTIMATION
        elif variant is not None and "3d" in str(variant).lower():
            task = ModelTask.CV_3D_DETECTION
        else:
            task = ModelTask.CV_OBJECT_DET
        return ModelInfo(
            model="centernet",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.CUSTOM,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the CenterNet model instance with default settings.

        Returns:
            Onnx model: The CenterNet model instance.
        """
        # Load model with defaults
        variant_name = kwargs.get("variant_name", "dla1x_od")
        path = f"test_files/onnx/centernet/{variant_name}.onnx"
        file = get_file(path)
        model = onnx.load(file)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the CenterNet model with default settings.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        variant_name = kwargs.get("variant_name", "dla1x_od")
        # Set input resolution based on the task:
        # Use 512x512 for Object Detection (OD) and Human Pose Estimation (HPE) variants,
        # and 1280x384 for 3D Bounding Box (3D_BB) Detection variants
        if "od" in variant_name or "hpe" in variant_name:
            h, w = 512, 512
        else:
            h, w = 1280, 384
        image_file = get_file(
            "https://github.com/xingyizhou/CenterNet/raw/master/images/17790319373_bd19b24cfc_k.jpg"
        )
        image = Image.open(image_file).convert("RGB").resize((h, w))
        m, s = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        preprocess = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=m, std=s)]
        )
        input_tensor = preprocess(image)
        inputs = input_tensor.unsqueeze(0)

        return inputs
