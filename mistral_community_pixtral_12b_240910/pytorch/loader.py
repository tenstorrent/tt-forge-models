# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Community Pixtral 12B 240910 model loader implementation for multimodal visual QA.

mistral-community/pixtral-12b-240910 is a community-provided checkpoint mirrored
from Mistral AI's torrent release of Pixtral-12B (v0.1, released 2024-09-10).
The processor is sourced from the Transformers-compatible sibling repo
(mistral-community/pixtral-12b) since the 240910 repo ships the raw Mistral
format (consolidated.safetensors, params.json, tekken.json).
"""

from typing import Optional

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Mistral Community Pixtral 12B 240910 model variants."""

    PIXTRAL_12B_240910 = "12B_240910"


class ModelLoader(ForgeModel):
    """Mistral Community Pixtral 12B 240910 model loader for multimodal visual QA."""

    _VARIANTS = {
        ModelVariant.PIXTRAL_12B_240910: ModelConfig(
            pretrained_model_name="mistral-community/pixtral-12b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PIXTRAL_12B_240910

    PROCESSOR_MODEL = "mistral-community/pixtral-12b"

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mistral Community Pixtral 12B 240910",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_MODEL)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
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
            conversation, add_generation_prompt=True
        )

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
