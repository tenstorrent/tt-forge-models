# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OFT (Orthographic Feature Transform) model loader implementation
"""
import torch

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from .src.oftnet import OftNet


class ModelLoader(ForgeModel):
    """OFT model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.grid_res = 0.5
        self.num_classes = 1
        self.frontend = "resnet18"
        self.topdown_layers = 8
        self.grid_height = 4.0

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'default'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "default"
        return ModelInfo(
            model="oft",
            variant=variant_name,
            group=ModelGroup.PRIORITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the OFT model instance with default settings.

        Returns:
            torch.nn.Module: The OFT model instance.
        """
        # Load model with defaults
        model = OftNet(
            num_classes=self.num_classes,
            frontend=self.frontend,
            topdown_layers=self.topdown_layers,
            grid_res=self.grid_res,
            grid_height=self.grid_height,
        )

        return model

    def load_inputs(self):
        """Load and return sample inputs for the OFT model with default settings.

        Returns:
            tuple: Sample inputs that can be fed to the model.
                - dummy_image (torch.Tensor): Input image tensor [B, 3, H, W]
                - dummy_calib (torch.Tensor): Camera calibration parameters [B, 3, 4]
                - dummy_grid (torch.Tensor): Bird's-eye view grid [B, D, W, 3]
        """
        # Fixed default parameters
        batch_size = 1
        grid_size = (80.0, 80.0)  # width, depth in meters

        # Create input tensors matching the original test
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        dummy_calib = torch.randn(batch_size, 3, 4)

        # Calculate grid dimensions
        grid_depth = int(grid_size[1] / self.grid_res)
        grid_width = int(grid_size[0] / self.grid_res)
        dummy_grid = torch.randn(batch_size, grid_depth, grid_width, 3)

        return (dummy_image, dummy_calib, dummy_grid)
