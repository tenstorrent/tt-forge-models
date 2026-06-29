#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video text-encoder component loader.

LongCat-Video (meituan-longcat/LongCat-Video) is a 13.6B text-to-video diffusion
pipeline. Its text encoder is a UMT5-XXL encoder (`UMT5EncoderModel`,
google/umt5-xxl), used to embed the prompt for cross-attention conditioning of
the DiT denoiser. This loader brings up that single encoder component.

Available variants:
- LONGCAT_VIDEO: meituan-longcat/LongCat-Video (text_encoder subfolder)
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

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

# The pipeline tokenizes the prompt to a fixed length of 512 (padding="max_length").
MAX_SEQUENCE_LENGTH = 512


class ModelVariant(StrEnum):
    """Available LongCat-Video text-encoder variants."""

    LONGCAT_VIDEO = "longcat_video"


class ModelLoader(ForgeModel):
    """LongCat-Video UMT5-XXL text-encoder loader."""

    _VARIANTS = {
        ModelVariant.LONGCAT_VIDEO: ModelConfig(
            pretrained_model_name="meituan-longcat/LongCat-Video",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LONGCAT_VIDEO

    DEFAULT_PROMPT = (
        "In a realistic photography style, a white boy around seven or eight "
        "years old sits on a park bench, holding an ice cream cone, beside a "
        "golden Labrador, on a sunny day with a green lawn and tall trees."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="longcat_video_text_encoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder="tokenizer",
            )
        return self._tokenizer

    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        """Load the UMT5-XXL text encoder."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = UMT5EncoderModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None):
        """Tokenize the default prompt to the pipeline's fixed 512-token length."""
        tokenizer = self._load_tokenizer()
        enc = tokenizer(
            [self.DEFAULT_PROMPT],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
        }
