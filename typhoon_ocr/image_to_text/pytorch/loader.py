# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Typhoon OCR model loader implementation for image-to-text OCR tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Typhoon OCR model variants for image-to-text tasks."""

    TYPHOON_OCR_3B = "3B"


class ModelLoader(ForgeModel):
    """Typhoon OCR model loader implementation for image-to-text OCR tasks."""

    _VARIANTS = {
        ModelVariant.TYPHOON_OCR_3B: LLMModelConfig(
            pretrained_model_name="typhoon-ai/typhoon-ocr-3b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TYPHOON_OCR_3B

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                },
                {
                    "type": "text",
                    "text": (
                        "Below is an image of a document page along with its dimensions. "
                        "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
                        "If the document contains images, use a placeholder like dummy.png for each image.\n"
                        "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
                        "RAW_TEXT_START\n\nRAW_TEXT_END"
                    ),
                },
            ],
        }
    ]

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._precomputed_inputs = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Typhoon OCR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )
        return self.processor

    def _prepare_raw_inputs(self):
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    @torch.no_grad()
    def _precompute_embeddings(self, model, raw_inputs):
        input_ids = raw_inputs["input_ids"]
        attention_mask = raw_inputs["attention_mask"]
        pixel_values = raw_inputs["pixel_values"]
        image_grid_thw = raw_inputs["image_grid_thw"]

        inner = model.model
        inputs_embeds = inner.get_input_embeddings()(input_ids)

        image_embeds = inner.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        ).pooler_output
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
            second_per_grid_ts=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        raw_inputs = self._prepare_raw_inputs()
        self._precomputed_inputs = self._precompute_embeddings(model, raw_inputs)

        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        inputs = self._precomputed_inputs

        if dtype_override is not None:
            inputs["inputs_embeds"] = inputs["inputs_embeds"].to(dtype_override)

        return inputs
