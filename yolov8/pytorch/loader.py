# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8 model loader — https://github.com/ultralytics/ultralytics

Uses the Ultralytics YOLOv8 implementation directly. The Detect head is set
to export mode so forward() returns a single (B, 84, N_anchors) tensor
instead of the non-traceable post-processing dict.

Dynamo config flags are set for fullgraph compilation because Ultralytics'
make_anchors uses data-dependent scalar ops (aten._local_scalar_dense).
"""

import torch
import torch._dynamo
from dataclasses import dataclass
from typing import Optional

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

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True


@dataclass
class YOLOv8Config(ModelConfig):
    resolution: int = 640


class ModelVariant(StrEnum):
    YOLOV8S_640 = "YOLOv8s_640"
    YOLOV8L_1280 = "YOLOv8l_1280"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.YOLOV8S_640: YOLOv8Config(
            pretrained_model_name="yolov8s",
            resolution=640,
        ),
        ModelVariant.YOLOV8L_1280: YOLOv8Config(
            pretrained_model_name="yolov8l",
            resolution=1280,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLOV8S_640

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOv8",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from ultralytics import YOLO

        variant = self._variant_config.pretrained_model_name
        yolo = YOLO(f"{variant}.pt")
        model = yolo.model.eval()
        # Export mode on Detect head: bypasses non-traceable post-processing
        # and returns a clean (B, num_classes+4, anchors) tensor.
        model.model[-1].export = True

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from torchvision import transforms
        from ...tools.utils import get_file

        cfg = self._variant_config
        image_path = str(get_file("https://ultralytics.com/images/bus.jpg"))

        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize((cfg.resolution, cfg.resolution)),
            transforms.ToTensor(),
        ])
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)
        return batch_tensor
