# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FastVLM model loader implementation for image-to-text tasks.
"""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


def prepare_multimodal_embeddings(model, input_ids, pixel_values):
    """Pre-compute multimodal embeddings by combining text and image features.

    This runs eagerly (outside torch.compile) to avoid data-dependent control
    flow in the model's prepare_inputs_labels_for_multimodal method.
    """
    image_features = model.encode_images(pixel_values)

    embed_tokens = model.get_model().embed_tokens

    batch_embeds = []
    cur_image_idx = 0
    for batch_idx in range(input_ids.shape[0]):
        cur_input_ids = input_ids[batch_idx]
        image_token_mask = cur_input_ids == IMAGE_TOKEN_INDEX
        image_token_indices = torch.where(image_token_mask)[0].tolist()

        if len(image_token_indices) == 0:
            batch_embeds.append(embed_tokens(cur_input_ids))
            continue

        segments = []
        # Before first image token
        if image_token_indices[0] > 0:
            segments.append(embed_tokens(cur_input_ids[: image_token_indices[0]]))

        for i, img_idx in enumerate(image_token_indices):
            segments.append(image_features[cur_image_idx])
            cur_image_idx += 1
            # Text between this image token and the next (or end)
            next_start = img_idx + 1
            next_end = (
                image_token_indices[i + 1]
                if i + 1 < len(image_token_indices)
                else len(cur_input_ids)
            )
            if next_start < next_end:
                segments.append(embed_tokens(cur_input_ids[next_start:next_end]))

        batch_embeds.append(torch.cat(segments, dim=0))

    return torch.stack(batch_embeds, dim=0)


class FastVLMWrapper(nn.Module):
    """Wrapper that accepts pre-computed inputs_embeds, bypassing multimodal prep."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        return Qwen2ForCausalLM.forward(
            self.model,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )


class ModelVariant(StrEnum):
    """Available FastVLM model variants for image-to-text tasks."""

    FAST_VLM_0_5B = "0.5B"


class ModelLoader(ForgeModel):
    """FastVLM model loader implementation for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FAST_VLM_0_5B: LLMModelConfig(
            pretrained_model_name="apple/FastVLM-0.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FAST_VLM_0_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="fast_vlm",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return a wrapped FastVLM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.raw_model = model

        return FastVLMWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FastVLM model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        # Build prompt with <image> placeholder using chat template
        messages = [
            {
                "role": "user",
                "content": "<image>\nDescribe this image in detail.",
            }
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Split around <image> and insert IMAGE_TOKEN_INDEX
        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tokenizer(
            pre, return_tensors="pt", add_special_tokens=False
        ).input_ids
        post_ids = self.tokenizer(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)

        # Load and process image through the vision tower
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.raw_model.get_vision_tower()
        pixel_values = vision_tower.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        # Pre-compute multimodal embeddings eagerly
        with torch.no_grad():
            inputs_embeds = prepare_multimodal_embeddings(
                self.raw_model, input_ids, pixel_values
            )

        return inputs_embeds
