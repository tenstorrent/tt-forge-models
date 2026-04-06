# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLO-World Small-640 ForgeModel loader.

Single variant only: Small_640 (640×640 input, YOLOv8-S backbone + YOLOWorldPAFPN neck).
Source: https://github.com/ailab-cvc/yolo-world
"""

from typing import Optional

import torch

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """YOLO-World Small-640 variant."""

    SMALL_640 = "Small_640"


class ModelLoader(ForgeModel):
    """ForgeModel loader for YOLO-World Small-640."""

    _VARIANTS = {
        ModelVariant.SMALL_640: ModelConfig(pretrained_model_name="Small_640"),
    }
    DEFAULT_VARIANT = ModelVariant.SMALL_640

    DEFAULT_TEXTS = "person,bus"

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        texts: Optional[str] = None,
    ):
        super().__init__(variant)
        self.texts = texts or self.DEFAULT_TEXTS
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLO-World-S640",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.model import build_yoloworld_s640

        checkpoint = str(
            get_file(f"test_files/pytorch/yoloworld/{self._variant_config.pretrained_model_name}.pth")
        )

        model = build_yoloworld_s640()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        texts = [[t.strip()] for t in self.texts.split(",")] + [[" "]]
        model.reparameterize(texts)
        model.eval()
        self._model = model
        return model

    def load_inputs(self, dtype_override=None, **kwargs):
        from .src.model import build_yoloworld_s640

        image_file = get_file("https://ultralytics.com/images/bus.jpg")

        import cv2
        import numpy as np

        img = cv2.imread(str(image_file))  # BGR uint8 HWC
        h, w = img.shape[:2]
        # Letterbox-resize to 640×640
        scale = min(640 / h, 640 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_h = (640 - new_h) // 2
        pad_w = (640 - new_w) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        # BGR→RGB, HWC→CHW, add batch dim
        img_rgb = canvas[:, :, ::-1].copy()
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        if dtype_override is not None:
            tensor = tensor.to(dtype_override)

        return tensor

    def post_process(self, output, output_dir=None):
        """Return raw (cls_logits, bbox_preds) tuple — post-processing is CPU-side."""
        return output
