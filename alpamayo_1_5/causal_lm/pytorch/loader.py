# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Alpamayo 1.5 model loader implementation for causal language modeling.

Note: Alpamayo-1.5-10B is a VLA (Vision-Language-Action) model that requires:
  - The proprietary alpamayo1_5 Python package (needs CUDA for flash-attn)
  - The gated nvidia/Cosmos-Reason2-8B model as its VLM backbone/tokenizer

Since these are unavailable in our environment, we fall back to loading
Qwen/Qwen3-VL-8B-Instruct, the public base model that Cosmos-Reason2-8B
and Alpamayo-1.5 are built on.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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

FALLBACK_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class ModelVariant(StrEnum):
    """Available Alpamayo 1.5 model variants for causal language modeling."""

    ALPAMAYO_1_5_10B = "1_5_10B"


class ModelLoader(ForgeModel):
    """Alpamayo 1.5 model loader implementation for causal language modeling tasks.

    Falls back to Qwen/Qwen3-VL-8B-Instruct (the public VLM backbone) since the
    full Alpamayo model requires proprietary alpamayo1_5 package and gated resources.
    """

    _VARIANTS = {
        ModelVariant.ALPAMAYO_1_5_10B: LLMModelConfig(
            pretrained_model_name="nvidia/Alpamayo-1.5-10B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALPAMAYO_1_5_10B

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Alpamayo 1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(FALLBACK_MODEL)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            FALLBACK_MODEL, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(FALLBACK_MODEL)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.sample_text}],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
