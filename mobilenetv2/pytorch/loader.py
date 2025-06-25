# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV2 model loader implementation
"""

from ...base import ForgeModel
import torch
from PIL import Image
from ...tools.utils import get_file
from torchvision import transforms
from .src.utils import download_model


class ModelLoader(ForgeModel):
    """Loads MobilenetV2 model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "mobilenet_v2"

    def load_model(self, dtype_override=None):
        """Load pretrained MobilenetV2 model."""
        model = download_model(
            torch.hub.load, "pytorch/vision:v0.10.0", self.model_name, pretrained=True
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for MobilenetV2 model"""

        # Get the Image
        image_file = get_file(
            "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
