# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet 50 torchvision model loader implementation for image classification
"""
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available ResNet 50 model variants."""

    TV = "torchvision"


class ModelLoader(ForgeModel):
    """ResNet 50 torchvision model loader implementation for image classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.TV: ModelConfig(
            pretrained_model_name="resnet50",
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
        self.weights = None

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
            model="resnet-50",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNet 50 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The ResNet 50 model instance for image classification.
        """
        # Load the ResNet-50 model with updated API
        self.weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=self.weights)

        # Convert to specified dtype
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ResNet 50 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor (preprocessed image) that can be fed to the model.
        """
        # Ensure weights are loaded (needed for transforms)
        if self.weights is None:
            self.weights = models.ResNet50_Weights.DEFAULT

        # Define a transformation to preprocess the input image
        preprocess = self.weights.transforms()

        # Load and preprocess the image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        img_t = preprocess(image)

        # Create batch tensor
        batch_t = torch.unsqueeze(img_t, 0)

        # Add additional batch dimensions if batch_size > 1
        if batch_size > 1:
            batch_t = batch_t.repeat(batch_size, 1, 1, 1)

        # Convert to specified dtype
        if dtype_override is not None:
            batch_t = batch_t.to(dtype_override)

        return batch_t

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable format.

        Args:
            outputs: Model output from a forward pass (logits)
                    Can be a single tensor or list of tensors (data parallel)

        Returns:
            str: Formatted output information with top 5 predictions
        """
        # Handle data parallel case (list of tensors)
        if isinstance(outputs, list):
            # Use the first device's output for decoding
            logits = outputs[0]
        else:
            # Handle single device case
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Get top 5 predictions
        _, indices = torch.topk(logits, 5)
        top5_predictions = indices[0].tolist()

        # Get predicted class and confidence
        predicted_class = torch.argmax(logits, dim=-1)
        confidence = torch.softmax(logits, dim=-1).max()

        return f"""
        ResNet 50 Output:
          - Top 5 predictions: {top5_predictions}
          - Predicted class: {predicted_class.item()}
          - Confidence: {confidence.item():.4f}
          - Output shape: {logits.shape}
        """
