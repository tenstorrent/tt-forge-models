# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
D-Fine model loader implementation for object detection
"""
import torch
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor
from ...tools.utils import get_file
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
    """Available D-Fine model variants."""

    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class ModelLoader(ForgeModel):
    """D-Fine model loader implementation for object detection tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.NANO: ModelConfig(
            pretrained_model_name="ustc-community/dfine-nano-coco",
        ),
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="ustc-community/dfine-small-coco",
        ),
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="ustc-community/dfine-medium-coco",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="ustc-community/dfine-large-coco",
        ),
        ModelVariant.XLARGE: ModelConfig(
            pretrained_model_name="ustc-community/dfine-xlarge-coco",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="d-fine",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """

        # Initialize processor
        self.processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the D-Fine model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The D-Fine model instance for object detection.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Load pre-trained model from HuggingFace
        model = DFineForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the D-Fine model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel values) that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        self.image = load_image(str(image_file))
        inputs = self.processor(images=self.image, return_tensors="pt")

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        # Add batch dimension if batch_size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
