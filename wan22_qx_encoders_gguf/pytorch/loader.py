#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAN 2.2 QX Encoders GGUF model loader implementation.

Loads the Rob1221rib/wan22-qx-encoders-gguf GGUF-quantized UMT5-XXL text
encoders used by the WAN 2.2 QX text-to-video generation pipeline. Multiple
quantization levels are provided, trading VRAM footprint for quality.

Available variants:
- Q3_K_S, Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0
"""

from typing import Any, Optional

import os

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel  # type: ignore[import]

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

REPO_ID = "Rob1221rib/wan22-qx-encoders-gguf"

# Base UMT5-XXL config source for architecture/tokenizer definition
UMT5_CONFIG = "google/umt5-xxl"


class ModelVariant(StrEnum):
    """Available WAN 2.2 QX UMT5-XXL GGUF text encoder variants."""

    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q3_K_S: "text_encoders/umt5-xxl-encoder-q3-k-s.gguf",
    ModelVariant.Q3_K_M: "text_encoders/umt5-xxl-encoder-q3-k-m.gguf",
    ModelVariant.Q4_K_S: "text_encoders/umt5-xxl-encoder-q4-k-s.gguf",
    ModelVariant.Q4_K_M: "text_encoders/umt5-xxl-encoder-q4-k-m.gguf",
    ModelVariant.Q5_K_S: "text_encoders/umt5-xxl-encoder-q5-k-s.gguf",
    ModelVariant.Q5_K_M: "text_encoders/umt5-xxl-encoder-q5-k-m.gguf",
    ModelVariant.Q6_K: "text_encoders/umt5-xxl-encoder-q6-k.gguf",
    ModelVariant.Q8_0: "text_encoders/umt5-xxl-encoder-q8-0.gguf",
}


class ModelLoader(ForgeModel):
    """WAN 2.2 QX Encoders GGUF model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=REPO_ID) for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    sample_text = (
        "A serene mountain landscape at sunset with flowing clouds "
        "over snow-capped peaks, cinematic lighting"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._encoder = None
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN22_QX_ENCODERS_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the UMT5-XXL text encoder from a GGUF checkpoint.

        Returns:
            UMT5EncoderModel instance loaded from the selected GGUF variant.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._encoder is None:
            gguf_path = hf_hub_download(REPO_ID, _GGUF_FILES[self._variant])
            config = UMT5Config.from_pretrained(UMT5_CONFIG)
            self._encoder = UMT5EncoderModel.from_pretrained(
                os.path.dirname(gguf_path),
                gguf_file=os.path.basename(gguf_path),
                config=config,
                torch_dtype=dtype,
            )
            self._encoder.eval()
        elif dtype_override is not None:
            self._encoder = self._encoder.to(dtype=dtype_override)
        return self._encoder

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(UMT5_CONFIG)
        return self._tokenizer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare tokenized text inputs for the UMT5-XXL text encoder.

        Returns:
            dict: Tokenized inputs with input_ids and attention_mask.
        """
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            self.sample_text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return inputs
