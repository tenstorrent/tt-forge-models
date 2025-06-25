# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Resnet model loader implementation for question answering
"""
import torch

import requests
from PIL import Image
from transformers import ResNetForImageClassification

from ...base import ForgeModel


class ModelLoader(ForgeModel):
    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "microsoft/resnet-50"
        self.input_shape = (3, 224, 224)

    def load_model(self, dtype_override=None):
        """Load a Resnet model from Hugging Face."""
        model = ResNetForImageClassification.from_pretrained(
            self.model_name, return_dict=False
        )

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Resnet models."""

        # Create a random input tensor with the correct shape, using default dtype
        inputs = torch.rand(1, *self.input_shape)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
