# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLO26 (Ultralytics/YOLO26) model loader implementation.

YOLO26 is the latest generation Ultralytics object detection model
available in five sizes (nano, small, medium, large, extra-large) trained
on the COCO dataset for 80-class detection.
"""
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchvision import transforms
from ultralytics import YOLO

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available YOLO26 model variants."""

    YOLO26N = "yolo26n"
    YOLO26S = "yolo26s"
    YOLO26M = "yolo26m"
    YOLO26L = "yolo26l"
    YOLO26X = "yolo26x"


class ModelLoader(ForgeModel):
    """YOLO26 model loader implementation."""

    _HF_REPO_ID = "Ultralytics/YOLO26"

    _VARIANTS = {
        ModelVariant.YOLO26N: ModelConfig(
            pretrained_model_name="yolo26n",
        ),
        ModelVariant.YOLO26S: ModelConfig(
            pretrained_model_name="yolo26s",
        ),
        ModelVariant.YOLO26M: ModelConfig(
            pretrained_model_name="yolo26m",
        ),
        ModelVariant.YOLO26L: ModelConfig(
            pretrained_model_name="yolo26l",
        ),
        ModelVariant.YOLO26X: ModelConfig(
            pretrained_model_name="yolo26x",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLO26N

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLO26",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        variant_name = self._variant_config.pretrained_model_name
        weights_path = hf_hub_download(
            repo_id=self._HF_REPO_ID, filename=f"{variant_name}.pt"
        )
        yolo_model = YOLO(weights_path)
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
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
