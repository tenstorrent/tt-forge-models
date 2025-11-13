# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv9 model loader implementation
"""
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from torch.hub import load_state_dict_from_url
from ultralytics.nn.tasks import DetectionModel
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from ...tools.utils import yolo_postprocess, VisionPreprocessor


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="yolov9",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    """YOLOv9 model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_variant = "yolov9c"
        self.model = None
        self._preprocessor = None

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv9 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv9 model instance.
        """

        variant = self.model_variant
        weights = load_state_dict_from_url(
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{variant}.pt",
            map_location="cpu",
        )
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        # Store model for potential use in input preprocessing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

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
            # YOLOv9 uses custom preprocessing: resize to 640x640 and ToTensor
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
                model_name="yolov9",
                custom_preprocess_fn=custom_preprocess_fn,
            )

        # If image is None, use default dataset image (backward compatibility)
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test[:1]")
            image = dataset[0]["image"]

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

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

    def output_postprocess(self, co_out):
        """Post-process YOLOv9 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv9 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)
