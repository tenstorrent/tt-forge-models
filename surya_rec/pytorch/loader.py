# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
import torch
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


class ModelVariant(StrEnum):
    """Available Surya Recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Surya Recognition model loader for OCR text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="vikp/surya_rec",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="surya_rec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import os

        # surya's settings.py calls torch_xla.devices() at import time which aborts the
        # process with SIGABRT in compile-only environments. Force CPU to skip that path.
        os.environ.setdefault("TORCH_DEVICE", "cpu")

        from surya.foundation import FoundationPredictor

        foundation_predictor = FoundationPredictor(device="cpu")
        self._processor = foundation_predictor.processor
        model = foundation_predictor.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        import os
        import numpy as np

        # Must be set before any surya import to prevent SIGABRT from torch_xla.devices()
        os.environ.setdefault("TORCH_DEVICE", "cpu")

        if not hasattr(self, "_processor") or self._processor is None:
            from surya.foundation import FoundationPredictor

            self._processor = FoundationPredictor(device="cpu").processor

        from surya.common.surya.schema import TaskNames

        dummy_image = np.zeros((196, 896, 3), dtype=np.uint8)
        batch_inputs = [
            {
                "task": TaskNames.ocr_without_boxes,
                "inputs": [
                    {"type": "image", "image": dummy_image, "rotated": False},
                    {"type": "text", "text": "", "math": False},
                ],
            }
        ] * batch_size

        processed = self._processor(batch_inputs, padding_side="left", device="cpu")

        input_ids = processed["input_ids"].to(dtype=torch.long)
        image_tiles = processed["image_tiles"]
        grid_thw = processed["grid_thw"].to(dtype=torch.long)
        attention_mask = processed["attention_mask"].to(dtype=torch.long)
        position_ids = processed["position_ids"].to(dtype=torch.long)

        if dtype_override is not None:
            image_tiles = image_tiles.to(dtype_override)

        return {
            "input_ids": input_ids,
            "image_tiles": image_tiles,
            "grid_thw": grid_thw,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
