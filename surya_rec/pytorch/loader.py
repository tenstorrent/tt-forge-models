# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Surya Recognition model loader implementation for OCR text recognition tasks.
"""
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
        from surya.common.surya.schema import TaskNames

        image = Image.new("RGB", (896, 196), color=(255, 255, 255))
        fp = self._foundation_predictor
        model = fp.model

        batch_input = fp.prepare_input(
            task_names=[TaskNames.ocr_without_boxes] * batch_size,
            images=[image] * batch_size,
            input_text=[None] * batch_size,
            math_modes=[False] * batch_size,
        )
        processed = fp.processor(batch_input, padding_side="left", device=model.device)

        input_ids = processed["input_ids"].to(dtype=torch.long)
        attention_mask = processed["attention_mask"].to(dtype=torch.long)
        position_ids = processed["position_ids"].to(dtype=torch.long)
        image_tiles = processed["image_tiles"].to(dtype=model.dtype)
        grid_thw = processed["grid_thw"].to(dtype=torch.long)

        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(
                pixel_values=image_tiles,
                grid_thw=grid_thw,
                encoder_chunk_size=4096,
                valid_batch_size=batch_size,
            )

        cache_position = (
            torch.arange(input_ids.shape[1], dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        if dtype_override is not None:
            image_embeddings = image_embeddings.to(dtype_override)

        return {
            "input_ids": input_ids,
            "image_embeddings": image_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
