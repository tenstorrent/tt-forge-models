# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PaliGemma model loader implementation for image-text-to-text generation.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

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
from ...tools.utils import get_file
from .src.model_utils import PaliGemmaWrapper


class ModelVariant(StrEnum):
    """Available PaliGemma model variants."""

    PALIGEMMA_3B_MIX_224 = "3B_Mix_224"


class ModelLoader(ForgeModel):
    """PaliGemma model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.PALIGEMMA_3B_MIX_224: ModelConfig(
            pretrained_model_name="fal/paligemma-3b-mix-224",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PALIGEMMA_3B_MIX_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PaliGemma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        model.eval()
        self._full_model = model

        if self.processor is None:
            self._load_processor()

        return PaliGemmaWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        prompt = "caption en"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        model = self._full_model
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            attention_mask = inputs.get("attention_mask")

            if input_ids.shape[0] < batch_size:
                input_ids = input_ids.repeat_interleave(batch_size, dim=0)
                pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
                if attention_mask is not None:
                    attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)

            if model.config.image_token_id >= model.config.text_config.vocab_size:
                llm_input_ids = input_ids.clone()
                llm_input_ids[input_ids == model.config.image_token_id] = 0
            else:
                llm_input_ids = input_ids

            inputs_embeds = model.get_input_embeddings()(llm_input_ids)

            image_features = model.model.get_image_features(
                pixel_values, return_dict=True
            ).pooler_output
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            special_image_mask = input_ids == model.config.image_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
                inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(1, seq_len + 1, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if dtype_override is not None:
            if inputs_embeds.dtype == torch.float32:
                inputs_embeds = inputs_embeds.to(dtype_override)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.processor is None:
            self._load_processor()

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.processor.decode(token_ids[0], skip_special_tokens=True)
