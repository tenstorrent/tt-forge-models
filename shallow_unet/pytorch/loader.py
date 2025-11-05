# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shallow uNet model loader implementation for image segmentation using Hugging Face AutoModel.
"""
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Shallow uNet model variants."""

    SHALLOW_UNET = "shallow_unet"


class ModelLoader(ForgeModel):
    """Shallow uNet model loader implementation for image segmentation tasks."""

    # Dictionary of available model variants
    # Using a lightweight segmentation model from HF Hub that works with AutoModel
    # For "shallow" UNet, we use a lighter-weight model compatible with AutoModelForImageSegmentation
    _VARIANTS = {
        ModelVariant.SHALLOW_UNET: ModelConfig(
            # Using a lightweight segmentation model that supports AutoModel
            # This can be replaced with any UNet-compatible model from HF Hub
            pretrained_model_name="briaai/RMBG-2.0",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SHALLOW_UNET

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
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="shallow_unet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
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
        """Load and return the Shallow uNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Shallow uNet model instance for image segmentation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Load pre-trained model from HuggingFace using AutoModel
        # Try AutoModelForImageSegmentation first, then fallback to generic AutoModel
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )
        except Exception:
            # Fallback: Try loading as generic AutoModel if AutoModelForImageSegmentation fails
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Shallow uNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict or torch.Tensor: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Process image using processor
        inputs = self.processor(images=image, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                # Convert the input dtype to dtype_override if specified
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        # If processor returns pixel_values, return that tensor directly for compatibility
        if "pixel_values" in inputs:
            return inputs["pixel_values"]
        else:
            # Otherwise return the full inputs dict
            return inputs

