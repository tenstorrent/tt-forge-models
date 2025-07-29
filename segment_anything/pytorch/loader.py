# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segment Anything (SAM2) model loader implementation
"""

import torch
from PIL import Image
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
from ...tools.utils import get_file
from typing import Optional


class ModelVariant(StrEnum):
    """Available Segment Anything model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Segment Anything model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/sam2-hiera-small",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="segment_anything",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_MASK_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "facebook/sam2-hiera-small"
        self.predictor = None

    def load_model(self, dtype_override=None):
        """Load a SAM2 model from Hugging Face."""

        # Currently model is skipped for the following reason:
        # Failed to install sam2. sam2 requires Python >=3.10.0 but the default version on Ubuntu 20.04 is 3.8. We found no other pytorch implementation of segment-anything.
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Create predictor
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)

        # Load and set the sample image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        self.predictor.set_image(image)

        return self.predictor

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for SAM2 model."""

        # Return the text prompt as in original test
        prompt = "Beautiful thing"
        return prompt
