# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr loader - vision tower component (dots_vit, NaViT-style image encoder).

This brings up the vision half of dots.ocr in isolation. ``dots_vit`` is a
42-layer NaViT-style transformer: a Conv2d patch embed (kernel=stride=14),
rotary position embeddings derived from the image grid, cu_seqlens-segmented
attention (eager O(n^2) path here, since flash-attn is unavailable), a SwiGLU
FFN, and a spatial patch merger (2x2). It produces merged patch embeddings that
the text decoder splices in at the image token.
"""
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoProcessor

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
from ..._common import DOTS_OCR_MODEL, DOTS_OCR_REVISION, build_demo_image
from .src.model import VisionTowerWrapper


class ModelVariant(StrEnum):
    """Available dots.ocr vision-tower variants."""

    BASE = "dots_vit"


class ModelLoader(ForgeModel):
    """Loader for the dots.ocr dots_vit vision tower."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=DOTS_OCR_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=DOTS_OCR_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        model.config.vision_config.attn_implementation = "eager"
        model.eval()
        return VisionTowerWrapper(model)

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        image = build_demo_image()
        inputs = self.processor(
            text=["<|img|><|imgpad|><|endofimg|>"],
            images=[image],
            return_tensors="pt",
        )
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return {
            "pixel_values": inputs["pixel_values"].to(dtype),
            "image_grid_thw": inputs["image_grid_thw"],
        }
