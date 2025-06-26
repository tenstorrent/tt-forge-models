# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG19-UNet model loader implementation
"""
import torch

from ...base import ForgeModel
from .src.vgg19_unet import VGG19UNet


class ModelLoader(ForgeModel):
    """VGG19-UNet model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.input_shape = (3, 512, 512)
        self.out_channels = 1

    def load_model(self, dtype_override=None):
        """Load and return the VGG19-UNet model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The VGG19-UNet model instance.
        """
        # Load model with defaults
        model = VGG19UNet(input_shape=self.input_shape, out_channels=self.out_channels)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the VGG19-UNet model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *self.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
