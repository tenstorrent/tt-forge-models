# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ovis2-1B HuggingFace-native model loader implementation for image-to-text tasks.
"""
from typing import Optional

import torch
from transformers import AutoProcessor, Ovis2ForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Ovis2-1B-hf model variants."""

    OVIS2_1B_HF = "1B"


class ModelLoader(ForgeModel):
    """Ovis2-1B-hf model loader for image-to-text tasks."""

    _VARIANTS = {
        ModelVariant.OVIS2_1B_HF: LLMModelConfig(
            pretrained_model_name="thisisiron/Ovis2-1B-hf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OVIS2_1B_HF

    sample_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_text = "Describe this image in detail."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ovis2-1B-hf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        if self.processor is None:
            self._load_processor()

        model = Ovis2ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.sample_image_url},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in (torch.long, torch.int):
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.processor.decode(next_token_id, skip_special_tokens=True)
