# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA model loader implementation for multimodal conditional generation.
"""

from typing import Optional

from PIL import Image
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
    """Available LLaVA model variants."""

    LLAVA_1_5_7B = "1.5_7B"
    LLAVA_1_5_13B = "1.5_13B"
    LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW = "llama-2-13b-chat-lightning-preview"


class ModelLoader(ForgeModel):
    """LLaVA model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_1_5_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-1.5-7b-hf",
        ),
        ModelVariant.LLAVA_1_5_13B: ModelConfig(
            pretrained_model_name="llava-hf/llava-1.5-13b-hf",
        ),
        ModelVariant.LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW: ModelConfig(
            pretrained_model_name="liuhaotian/llava-llama-2-13b-chat-lightning-preview",
        ),
    }

    _PROCESSOR_OVERRIDES = {
        ModelVariant.LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW: "llava-hf/llava-1.5-13b-hf",
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_1_5_7B

    sample_image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    sample_text = "What’s shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        variant_groups = {
            ModelVariant.LLAVA_1_5_13B: ModelGroup.VULCAN,
            ModelVariant.LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW: ModelGroup.VULCAN,
        }

        return ModelInfo(
            model="LLaVA",
            variant=variant,
            group=variant_groups.get(variant, ModelGroup.RED),
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        processor_name = self._PROCESSOR_OVERRIDES.get(
            self._variant, self._variant_config.pretrained_model_name
        )
        self.processor = AutoProcessor.from_pretrained(processor_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA model instance."""
        model_name = self._variant_config.pretrained_model_name
        if self._variant == ModelVariant.LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW:
            kwargs.setdefault("ignore_mismatched_sizes", True)
        model = LlavaForConditionalGeneration.from_pretrained(str(model_name), **kwargs)

        if self._variant == ModelVariant.LLAVA_LLAMA2_13B_CHAT_LIGHTNING_PREVIEW:
            # The checkpoint has vocab_size=32000 but the image token index is also 32000,
            # making it out of range. Resize embeddings to accommodate the image token.
            image_token_index = model.config.image_token_index
            current_vocab_size = model.config.text_config.vocab_size
            if image_token_index >= current_vocab_size:
                model.resize_token_embeddings(image_token_index + 1)

        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA."""
        if self.processor is None:
            self._load_processor()

        # Build prompt
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

        image = Image.new("RGB", (336, 336), color=(128, 128, 128))

        # Preprocess
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
