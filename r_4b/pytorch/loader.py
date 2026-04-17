# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
R-4B model loader implementation for multimodal conditional generation.
"""

from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

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


class R4BLanguageModelWrapper(nn.Module):
    """Wrapper that runs only the language model + lm_head.

    Vision processing uses dynamic Python control flow (image_sizes,
    grid_pinpoints, etc.) that is incompatible with torch.compile, so
    we pre-compute inputs_embeds in load_inputs and only compile this
    text-only portion.
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


class ModelVariant(StrEnum):
    """Available R-4B model variants."""

    R_4B = "R_4B"


class ModelLoader(ForgeModel):
    """R-4B model loader for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.R_4B: ModelConfig(
            pretrained_model_name="YannQi/R-4B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.R_4B

    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize R-4B model loader."""
        super().__init__(variant)
        self.processor = None
        self.raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="R-4B",
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
        """Load and return the R-4B model wrapped for text-only forward pass."""
        model_name = self._variant_config.pretrained_model_name

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tie_word_embeddings = False

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype_override if dtype_override else torch.float32,
            "config": config,
        }
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        self.raw_model = model
        return R4BLanguageModelWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return pre-computed inputs_embeds for R-4B.

        Vision features are computed eagerly here so that only the
        language model (which is torch.compile-friendly) is compiled.
        """
        if self.raw_model is None:
            self.load_model(dtype_override=dtype_override)
        if self.processor is None:
            self._load_processor()

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                    },
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        rmodel = self.raw_model.model
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            image_sizes = inputs["image_sizes"]
            attention_mask = inputs["attention_mask"]

            inputs_embeds = rmodel.get_input_embeddings()(input_ids)

            image_features = rmodel.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self.raw_model.config.vision_feature_layer,
                vision_feature_select_strategy=self.raw_model.config.vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0)

            special_image_mask = (
                (input_ids == rmodel.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        if dtype_override:
            inputs_embeds = cast_input_to_type(inputs_embeds, dtype_override)
            attention_mask = cast_input_to_type(attention_mask, dtype_override)

        return {
            "inputs_embeds": inputs_embeds.clone().detach(),
            "attention_mask": attention_mask.clone().detach(),
        }
