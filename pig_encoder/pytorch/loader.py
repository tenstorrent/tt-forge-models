#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pig Encoder model loader implementation.

Loads the calcuis/pig-encoder GGUF-format text encoder models. These are
standard CLIP text encoders (e.g. openai/clip-vit-large-patch14) distributed
in GGUF format for efficient storage and inference in ComfyUI workflows.

Available variants:
- CLIP_L_F16: CLIP-L text encoder in fp16 GGUF format
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]
from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore[import]

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

REPO_ID = "calcuis/pig-encoder"

# Variant-specific GGUF filenames
CLIP_L_FILENAME = "clip_l_fp32-f16.gguf"

# CLIP-L base config source for architecture definition
CLIP_L_CONFIG = "openai/clip-vit-large-patch14"


class ModelVariant(StrEnum):
    """Available Pig Encoder model variants."""

    CLIP_L_F16 = "clip_l_f16"


class ModelLoader(ForgeModel):
    """Pig Encoder model loader for GGUF-format text encoder models."""

    _VARIANTS = {
        ModelVariant.CLIP_L_F16: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.CLIP_L_F16

    sample_text = "a pinky pig moving quickly in a beautiful winter scenery"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._encoder = None
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PIG_ENCODER",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Pig Encoder model from GGUF format.

        Returns:
            CLIPTextModel instance loaded from GGUF checkpoint.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._encoder is None:
            gguf_path = hf_hub_download(REPO_ID, CLIP_L_FILENAME)
            self._encoder = CLIPTextModel.from_pretrained(
                CLIP_L_CONFIG,
                gguf_file=gguf_path,
                torch_dtype=dtype,
            )
            self._encoder.eval()
        elif dtype_override is not None:
            self._encoder = self._encoder.to(dtype=dtype_override)
        return self._encoder

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = CLIPTokenizer.from_pretrained(CLIP_L_CONFIG)
        return self._tokenizer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare tokenized text inputs for the CLIP text encoder.

        Returns:
            dict: Tokenized inputs with input_ids and attention_mask.
        """
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            self.sample_text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs
