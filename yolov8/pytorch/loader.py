# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 model loader implementation
"""
import torch
from typing import Optional
from PIL import Image
from torchvision import transforms
from datasets import load_dataset

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
from ...tools.utils import VisionPreprocessor
from ultralytics import YOLO


class ModelVariant(StrEnum):
    """Available YOLOv8 model variants."""

    YOLOV8X = "yolov8x"
    YOLOV8N = "yolov8n"


class ModelLoader(ForgeModel):
    """YOLOv8 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV8X: ModelConfig(
            pretrained_model_name="yolov8x",
        ),
        ModelVariant.YOLOV8N: ModelConfig(
            pretrained_model_name="yolov8n",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV8X

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [ModelVariant.YOLOV8X]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="yolov8",
            variant=variant,
            group=group,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv8 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv8 model instance.
        """
        if self.model is None:
            # Get the model name from the instance's variant config
            model_name = self._variant_config.pretrained_model_name
            
            # Load YOLOv8 model using ultralytics
            yolo_model = YOLO(f"{model_name}.pt")
            # Get the underlying PyTorch model
            model = yolo_model.model
            model.eval()
            
            # Store model for potential use in preprocessing
            self.model = model
            
            # Update preprocessor with cached model if it exists
            if self._preprocessor is not None:
                self._preprocessor.set_cached_model(model)

            # Only convert dtype if explicitly requested
            if dtype_override is not None:
                model = model.to(dtype_override)
                self.model = model

        return self.model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default dataset image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            # YOLOv8 uses custom preprocessing: resize to 640x640 and ToTensor
            def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                transform = transforms.Compose(
                    [
                        transforms.Resize((640, 640)),
                        transforms.ToTensor(),
                    ]
                )
                return transform(img)

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name="yolov8",
                custom_preprocess_fn=custom_preprocess_fn,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        # If image is None, use huggingface cats-image dataset (backward compatibility)
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test[:1]")
            image = dataset[0]["image"]

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            torch.Tensor: Post-processed output tensor (raw output for YOLOv8).
        """
        # YOLOv8 outputs are already in the correct format
        # Return the output as-is (can be extended with NMS if needed)
        return output
