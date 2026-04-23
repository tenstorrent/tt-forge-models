# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer B2 Human Parse model loader implementation
"""

import numpy as np
import PIL.Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Segformer B2 Human Parse model variants."""

    SEGFORMER_B2_HUMAN_PARSE_24 = "segformer-b2-human-parse-24"


class ModelLoader(ForgeModel):
    """Segformer B2 Human Parse model loader implementation for human parsing semantic segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SEGFORMER_B2_HUMAN_PARSE_24: ModelConfig(
            pretrained_model_name="yolo12138/segformer-b2-human-parse-24",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SEGFORMER_B2_HUMAN_PARSE_24

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SegformerB2HumanParse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = SegformerImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Segformer B2 Human Parse model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Segformer B2 Human Parse model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Segformer B2 Human Parse model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel_values) that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = PIL.Image.fromarray(
            np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        )
        inputs = self.processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
