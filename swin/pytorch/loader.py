# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin model loader implementation
"""

from ...base import ForgeModel
from torchvision import models
import torch
from PIL import Image
from torchvision import models
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Loads Swin model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "swin_t"
        self.weight_name = "Swin_T_Weights"
        self._weights = None

    def load_model(self, dtype_override=None):
        """Load pretrained Swin model."""
        weights = getattr(models, self.weight_name).DEFAULT
        model = getattr(models, self.model_name)(weights=weights)
        model.eval()

        self._weights = weights

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Swin model"""

        preprocess = self._weights.transforms()
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        inputs = batch_t.contiguous()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
