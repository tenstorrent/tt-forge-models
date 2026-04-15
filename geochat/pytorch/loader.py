# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GeoChat model loader implementation for multimodal visual question answering.

GeoChat is a fine-tuned LLaVA-1.5-7B model for geospatial tasks. The original
HuggingFace repo (MBZUAI/geochat-7B) uses a custom ``model_type: "geochat"``
that is not registered in the transformers Auto mappings and ships no custom
modeling code.  We therefore load via the architecturally-identical
``llava-hf/llava-1.5-7b-hf`` checkpoint, which uses the native HF LLaVA
implementation and works seamlessly with the random-weights compile-only flow.
"""

from typing import Optional

from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor

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


class ModelVariant(StrEnum):
    """Available GeoChat model variants."""

    GEOCHAT_7B = "7B"


class ModelLoader(ForgeModel):
    """GeoChat model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.GEOCHAT_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-1.5-7b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEOCHAT_7B

    sample_text = "What do you see in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize GeoChat model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GeoChat",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GeoChat model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlavaForConditionalGeneration.from_pretrained(
            str(model_name), **model_kwargs
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for GeoChat."""
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, padding=True, add_generation_prompt=True
        )

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
