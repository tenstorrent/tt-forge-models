# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BAMBI Models (cpraschl/bambi-models) model loader implementation.

YOLOv8x-based object detection models fine-tuned for wildlife detection
in RGB and thermal drone imagery.
"""
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
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


class ModelVariant(StrEnum):
    """Available BAMBI Models variants."""

    RGB = "rgb"
    MATCHED_RGB = "matched_rgb"
    MATCHED_THERMAL = "matched_thermal"
    THERMAL_MERGED = "thermal_merged"
    THERMAL_ORIGINAL = "thermal_original"


class ModelLoader(ForgeModel):
    """BAMBI Models loader implementation."""

    _VARIANTS = {
        ModelVariant.RGB: ModelConfig(
            pretrained_model_name="rgb/weights/best.pt",
        ),
        ModelVariant.MATCHED_RGB: ModelConfig(
            pretrained_model_name="matched_rgb/weights/best.pt",
        ),
        ModelVariant.MATCHED_THERMAL: ModelConfig(
            pretrained_model_name="matched_thermal/weights/best.pt",
        ),
        ModelVariant.THERMAL_MERGED: ModelConfig(
            pretrained_model_name="thermal_merged/weights/best.pt",
        ),
        ModelVariant.THERMAL_ORIGINAL: ModelConfig(
            pretrained_model_name="thermal_original/weights/best.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RGB

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BAMBI Models",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("cpraschl/bambi-models", filename)
        yolo_model = YOLO(model_path)
        model = yolo_model.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
