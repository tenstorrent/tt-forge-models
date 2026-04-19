# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-NeXT (v1.6) model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
from ...tools.utils import get_file, cast_input_to_type


class LlavaNextWrapper(torch.nn.Module):
    """Wrapper that bypasses image processing in the compiled graph.

    LlavaNext's image processing pipeline uses numpy operations and
    data-dependent control flow (image_sizes, np.prod, modulo on shapes)
    that are incompatible with torch.compile/dynamo. This wrapper accepts
    pre-computed inputs_embeds so only the language model runs compiled.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )


class ModelVariant(StrEnum):
    """Available LLaVA-NeXT model variants."""

    LLAVA_V1_6_MISTRAL_7B = "v1.6_Mistral_7B"
    LLAVA_V1_6_VICUNA_7B = "v1.6_Vicuna_7B"


class ModelLoader(ForgeModel):
    """LLaVA-NeXT model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.LLAVA_V1_6_MISTRAL_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        ),
        ModelVariant.LLAVA_V1_6_VICUNA_7B: ModelConfig(
            pretrained_model_name="llava-hf/llava-v1.6-vicuna-7b-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAVA_V1_6_MISTRAL_7B

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize LLaVA-NeXT model loader."""
        super().__init__(variant)
        self.processor = None
        self.wrapper = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LLaVA-NeXT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LLaVA-NeXT model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = LlavaNextForConditionalGeneration.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        self.wrapper = LlavaNextWrapper(model)
        return self.wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-NeXT.

        Pre-computes inputs_embeds eagerly on CPU so that the image
        processing pipeline (which uses numpy ops and data-dependent
        control flow) is not traced by torch.compile.
        """
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

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs["image_sizes"]

        if dtype_override:
            pixel_values = cast_input_to_type(pixel_values, dtype_override)

        inner_model = self.wrapper.model.model
        with torch.no_grad():
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)
            if dtype_override:
                inputs_embeds = inputs_embeds.to(dtype_override)

            image_features = inner_model.get_image_features(
                pixel_values,
                image_sizes,
                return_dict=True,
            ).pooler_output
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            special_image_mask = inner_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
