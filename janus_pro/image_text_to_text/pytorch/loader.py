# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro image-text-to-text (understanding) loader.

Brings up the multimodal *understanding* forward pass of DeepSeek Janus-Pro
(image + text -> text logits) via the transformers-native
``JanusForConditionalGeneration``. This complements the existing ``text_to_image``
component loader, which targets the image-generation path.

The weights are the transformers-format conversion of ``deepseek-ai/Janus-Pro-7B``,
published as ``deepseek-community/Janus-Pro-7B`` (identical weights, model_type
``janus``). The original ``deepseek-ai`` checkpoint uses the custom
``multi_modality`` architecture and is not loadable through the standard Auto
classes.
"""

from typing import Optional

import torch
from transformers import AutoProcessor, JanusForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available Janus-Pro understanding variants (transformers-format weights)."""

    PRO_1B = "1B"
    PRO_7B = "7B"


class ModelLoader(ForgeModel):
    """Janus-Pro image-text-to-text (understanding) loader."""

    _VARIANTS = {
        ModelVariant.PRO_1B: ModelConfig(
            pretrained_model_name="deepseek-community/Janus-Pro-1B",
        ),
        ModelVariant.PRO_7B: ModelConfig(
            pretrained_model_name="deepseek-community/Janus-Pro-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRO_7B

    sample_image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Janus-Pro understanding loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus-Pro",
            variant=variant,
            group=ModelGroup.GENERALITY,
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
        """Load and return the Janus-Pro understanding model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = JanusForConditionalGeneration.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample image+text inputs for the understanding forward pass."""
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

        from PIL import Image

        image = Image.open(str(get_file(self.sample_image))).convert("RGB")

        inputs = self.processor(
            images=[image] * batch_size,
            text=[text_prompt] * batch_size,
            return_tensors="pt",
        )

        # generation_mode is a processor bookkeeping key, not a model forward arg.
        inputs.pop("generation_mode", None)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
