# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bee model loader implementation for multimodal conditional generation.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional

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
from ...tools.utils import cast_input_to_type, get_file


class BeeLanguageModelWrapper(nn.Module):
    """Wrapper that runs only the language model head of Bee.

    Vision features are pre-computed on CPU and merged into inputs_embeds
    before this wrapper is called, avoiding torch.compile issues with
    the vision encoder's data-dependent control flow.
    """

    def __init__(self, bee_model):
        super().__init__()
        self.language_model = bee_model.model.language_model
        self.lm_head = bee_model.lm_head

    def forward(self, inputs_embeds, attention_mask=None):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available Bee model variants."""

    BEE_8B_RL = "Bee_8B_RL"


class ModelLoader(ForgeModel):
    """Bee model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.BEE_8B_RL: ModelConfig(
            pretrained_model_name="Open-Bee/Bee-8B-RL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEE_8B_RL

    sample_text = "What are these?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Bee model loader."""
        super().__init__(variant)
        self.processor = None
        self.raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bee",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Bee and return a language-model-only wrapper."""
        model_name = self._variant_config.pretrained_model_name
        kwargs.setdefault("trust_remote_code", True)

        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        self.raw_model = model
        return BeeLanguageModelWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Pre-compute inputs_embeds with vision features merged on CPU."""
        if self.raw_model is None:
            self.load_model(dtype_override=dtype_override)

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

        with torch.no_grad():
            model = self.raw_model
            inner = model.model

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            image_sizes = inputs["image_sizes"]
            batch_num_images = inputs.get("batch_num_images")

            inputs_embeds = inner.get_input_embeddings()(input_ids)

            image_features = inner.get_image_features(
                pixel_values,
                image_sizes,
                batch_num_images=batch_num_images,
            )
            image_features = torch.cat(image_features, dim=0)

            special_image_mask = (input_ids == model.config.image_token_id).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds)
            image_features = image_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": inputs["attention_mask"],
        }

        if dtype_override:
            for key in result:
                result[key] = cast_input_to_type(result[key], dtype_override)

        return result
