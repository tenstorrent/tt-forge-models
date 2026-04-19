# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LLaVA-NeXT (v1.6) model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
import torch.nn as nn
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


class LlavaNextLanguageModelWrapper(nn.Module):
    """Wrapper that runs only the language model and lm_head.

    LlavaNextForConditionalGeneration uses image_sizes for Python-level
    arithmetic inside get_image_features, which fails under torch.compile
    with XLA backend. This wrapper accepts pre-computed inputs_embeds
    (with image features already merged) so the vision pipeline runs
    entirely on CPU before compilation.
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits


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
        self.full_model = None

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

        self.full_model = model
        return LlavaNextLanguageModelWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for LLaVA-NeXT."""
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

        with torch.no_grad():
            inner_model = self.full_model.model
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

            image_features = inner_model.get_image_features(
                pixel_values, image_sizes
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

        inputs_embeds = inputs_embeds.clone().detach()

        if dtype_override:
            inputs_embeds = cast_input_to_type(inputs_embeds, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
