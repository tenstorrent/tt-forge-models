# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL-4B-Instruct-action model loader implementation for image to text.

The Qwen3-VL vision encoder has extensive data-dependent control flow
(grid_thw values determine loop counts and tensor shapes). In compile-only
mode, all tensor values on XLA are zero-initialized, making the vision encoder
unusable. This loader precomputes inputs_embeds (with vision features) on CPU
so that only the language model portion is compiled for TT XLA.
"""

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


SAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


class ModelVariant(StrEnum):
    """Available Qwen3-VL-4B-Instruct-action model variants."""

    QWEN3_VL_4B_INSTRUCT_ACTION = "4b_instruct_action"


class ModelLoader(ForgeModel):
    """Qwen3-VL-4B-Instruct-action model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION: LLMModelConfig(
            pretrained_model_name="229nagibator229/Qwen3-VL-4B-Instruct-action",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_4B_INSTRUCT_ACTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._model_ref = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="qwen3_vl_4b_instruct_action",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, dtype="auto", device_map="auto", **model_kwargs
        )
        model.eval()
        self._model_ref = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        raw_inputs = self.processor.apply_chat_template(
            SAMPLE_MESSAGES,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Precompute inputs_embeds with vision features on CPU.
        # This bypasses the vision encoder on XLA where integer tensor
        # values (grid_thw) are zero-initialized in compile-only mode.
        model = self._model_ref
        inner = model.model

        with torch.no_grad():
            input_ids = raw_inputs["input_ids"]
            pixel_values = raw_inputs["pixel_values"]
            image_grid_thw = raw_inputs["image_grid_thw"]
            attention_mask = raw_inputs["attention_mask"]

            inputs_embeds = inner.get_input_embeddings()(input_ids)

            image_outputs = inner.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = inner.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            position_ids = inner.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
            )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
