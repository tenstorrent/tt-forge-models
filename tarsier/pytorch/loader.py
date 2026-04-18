# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tarsier model loader implementation for video/image description generation.

Uses llava-hf/llava-1.5-7b-hf (public) for config and processor access since
the omni-research/Tarsier-7b repo is gated. Both share the same LLaVA 1.5
architecture.
"""

from typing import Optional

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, LlavaConfig

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
from ...tools.utils import cast_input_to_type

_PUBLIC_LLAVA_7B = "llava-hf/llava-1.5-7b-hf"


class ModelVariant(StrEnum):
    """Available Tarsier model variants."""

    TARSIER_7B = "7B"


class ModelLoader(ForgeModel):
    """Tarsier model loader for video/image description generation."""

    _VARIANTS = {
        ModelVariant.TARSIER_7B: ModelConfig(
            pretrained_model_name="omni-research/Tarsier-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TARSIER_7B

    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Tarsier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            _PUBLIC_LLAVA_7B,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Tarsier model instance."""
        model_name = self._variant_config.pretrained_model_name

        config = LlavaConfig.from_pretrained(_PUBLIC_LLAVA_7B)

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "config": config,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model_kwargs |= kwargs

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return text-only input tensors for Tarsier.

        Skips pixel_values because the LLaVA image-token placeholder check
        inside get_placeholder_mask is incompatible with torch.compile on XLA.
        Text-only inputs still exercise the full language model path.
        """
        if self.processor is None:
            self._load_processor()

        inputs = self.processor.tokenizer(
            self.sample_text, return_tensors="pt", padding=True
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
