# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3 model loader implementation for vision-language tasks.

The HF variant `OpenGVLab/InternVL3-1B-hf` is the smallest public InternVL3
checkpoint exposed via the standard `transformers` `InternVLForConditionalGeneration`
interface (no `trust_remote_code` needed). It pairs an InternViT vision encoder
with a Qwen2-based text decoder (~938M params total).
"""
import torch
from transformers import InternVLForConditionalGeneration, AutoProcessor
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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available InternVL3 model variants."""

    INTERNVL3_1B_HF = "InternVL3_1B_hf"


class ModelLoader(ForgeModel):
    """InternVL3 model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.INTERNVL3_1B_HF: LLMModelConfig(
            pretrained_model_name="OpenGVLab/InternVL3-1B-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INTERNVL3_1B_HF

    # Standard demo image used by Idefics3/SmolVLM examples — single-image, single-turn.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
                },
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="InternVL3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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
        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = InternVLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.eval()
        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Wrapper.forward expects exactly these three keys; drop everything else
        # so the runner doesn't try to feed extra kwargs the wrapper rejects.
        keep = ("input_ids", "attention_mask", "pixel_values")
        return {k: inputs[k] for k in keep if k in inputs}
