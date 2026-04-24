# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
import os
import numpy as np
import torch
from PIL import Image
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

# surya/settings.py calls torch_xla.devices() at import time which crashes on TT hardware.
# Setting TORCH_DEVICE=cpu short-circuits that path before any surya import occurs.
os.environ.setdefault("TORCH_DEVICE", "cpu")


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
        self._foundation_predictor = None

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
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        self._foundation_predictor = FoundationPredictor(device="cpu")
        RecognitionPredictor(self._foundation_predictor)
        model = self._foundation_predictor.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        if self._foundation_predictor is None:
            self._foundation_predictor = FoundationPredictor(device="cpu")
            RecognitionPredictor(self._foundation_predictor)

        image = Image.new("RGB", (896, 196), color=(255, 255, 255))
        image_np = np.array(image)

        batch = self._foundation_predictor.prepare_input(
            ["ocr_without_boxes"], [image_np], [None], [False]
        )
        processed = self._foundation_predictor.processor(
            batch, device=torch.device("cpu")
        )

        inputs = {
            "input_ids": processed["input_ids"],
            "image_tiles": processed["image_tiles"],
            "attention_mask": processed["attention_mask"],
            "position_ids": processed["position_ids"],
            "grid_thw": processed["grid_thw"],
            # get_logits asserts hidden_states.shape[1] == 1; logits_to_keep=1
            # slices to the last token before that assertion runs.
            "logits_to_keep": 1,
        }

        if dtype_override is not None:
            inputs["image_tiles"] = inputs["image_tiles"].to(dtype_override)

        return inputs
