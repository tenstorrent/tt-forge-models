# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nougat model loader implementation for document understanding tasks.
"""
import torch
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Nougat model variants for document understanding tasks."""

    NOUGAT_BASE = "nougat_base"
    NOUGAT_SMALL = "nougat_small"
    NOUGAT_LATEX_BASE = "nougat_latex_base"


class ModelLoader(ForgeModel):
    """Nougat model loader implementation for document understanding tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NOUGAT_BASE: ModelConfig(
            pretrained_model_name="facebook/nougat-base",
        ),
        ModelVariant.NOUGAT_SMALL: ModelConfig(
            pretrained_model_name="facebook/nougat-small",
        ),
        ModelVariant.NOUGAT_LATEX_BASE: ModelConfig(
            pretrained_model_name="Norm/nougat-latex-base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NOUGAT_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="nougat",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = NougatProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = Image.new("RGB", (896, 1120), color=(255, 255, 255))
        # Call image_processor directly to avoid NougatProcessor forwarding None bool fields,
        # which triggers strict validation errors in newer huggingface_hub.
        pixel_values = self.processor.image_processor(
            image, return_tensors="pt"
        ).pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        # Add batch dimension
        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        bos_token_id = self.processor.tokenizer.bos_token_id
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long)

        return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output
