# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv11 model loader implementation
"""
from typing import Optional

from datasets import load_dataset
from torchvision import transforms
from ultralytics import YOLO

from ...base import ForgeModel

from ...config import (
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
    Framework,
)
from ...tools.utils import yolo_postprocess


class ModelVariant(StrEnum):
    """Available YOLOv11 model variants."""

    YOLOV11N = "yolo11n"
    YOLOV11S = "yolo11s"
    YOLOV11M = "yolo11m"
    YOLOV11L = "yolo11l"
    YOLOV11X = "yolo11x"


class ModelLoader(ForgeModel):
    """YOLOv11 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV11N: ModelConfig(
            pretrained_model_name="yolo11n",
        ),
        ModelVariant.YOLOV11S: ModelConfig(
            pretrained_model_name="yolo11s",
        ),
        ModelVariant.YOLOV11M: ModelConfig(
            pretrained_model_name="yolo11m",
        ),
        ModelVariant.YOLOV11L: ModelConfig(
            pretrained_model_name="yolo11l",
        ),
        ModelVariant.YOLOV11X: ModelConfig(
            pretrained_model_name="yolo11x",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV11X

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
            model="yolo11",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv11 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv11 model instance.
        """
        # Get the model name from the instance's variant config
        variant = self._variant_config.pretrained_model_name

        yolo_wrapper = YOLO(f"{variant}.pt")
        model = yolo_wrapper.model  # ultralytics.nn.tasks.DetectionModel
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOv11 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        image_tensor = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        batch_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor

    def post_process(self, co_out):
        """Post-process YOLOv11 model outputs to extract detection results.

        Args:
            co_out: Raw model output tensor from YOLOv11 forward pass.

        Returns:
            Post-processed detection results.
        """
        return yolo_postprocess(co_out)
