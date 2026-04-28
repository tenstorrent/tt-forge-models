# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Manga OCR model loader implementation for image-to-text OCR tasks.
"""
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
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
    """Available Manga OCR model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Manga OCR model loader implementation for Japanese text recognition."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="kha-white/manga-ocr-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.decoder_start_token_id = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="manga_ocr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)
        self.decoder_start_token_id = model.config.decoder_start_token_id

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name)

        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        start_token = self.decoder_start_token_id if self.decoder_start_token_id is not None else 2
        decoder_input_ids = torch.full((batch_size, 1), start_token, dtype=torch.long)

        return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}
