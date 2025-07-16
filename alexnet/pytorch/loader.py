# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation
"""

import torch
from typing import Optional
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel

from PIL import Image
from ...tools.utils import get_file, print_compiled_model_results
from torchvision import transforms


class Variant(StrEnum):
    """Available AlexNet model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Loads AlexNet model and sample input."""
    
    # Dictionary of available model variants
    _VARIANTS = {
        Variant.BASE: ModelConfig(
            pretrained_model_name="pytorch/vision:v0.10.0",
        )
    }

    DEFAULT_VARIANT = Variant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[StrEnum] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional StrEnum specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    def load_model(self, dtype_override=None):
        """Load pretrained AlexNet model."""

        model_name = "alexnet"
        model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for AlexNet model"""

        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
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

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
