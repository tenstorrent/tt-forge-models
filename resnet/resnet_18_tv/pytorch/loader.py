# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet 18 torchvision model loader implementation for image classification
"""
import torch
import torchvision
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ResNet 18 model variants."""

    TV = "torchvision"


class ModelLoader(ForgeModel):
    """ResNet 18 torchvision model loader implementation for image classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.TV: ModelConfig(
            pretrained_model_name="resnet18",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TV

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.input_shape = (3, 224, 224)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="resnet-18",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNet 18 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The ResNet 18 model instance for image classification.
        """
        # Get the model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load pre-trained model from torchvision
        model = torchvision.models.get_model(model_name, pretrained=True)

        # Convert to specified dtype
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ResNet 18 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor (pixel values) that can be fed to the model.
        """
        rand_kwargs = {}
        if dtype_override is not None:
            rand_kwargs["dtype"] = dtype_override

        # Generate random input tensor
        inputs = torch.rand((batch_size, *self.input_shape), **rand_kwargs)

        # Ensure dtype conversion
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable format.

        Args:
            outputs: Model output from a forward pass (logits)

        Returns:
            str: Formatted output information
        """
        # Get logits
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Get predicted class
        predicted_class = torch.argmax(logits, dim=-1)
        confidence = torch.softmax(logits, dim=-1).max()

        return f"""
        ResNet 18 Output:
          - Predicted class: {predicted_class.item()}
          - Confidence: {confidence.item():.4f}
          - Output shape: {logits.shape}
        """
