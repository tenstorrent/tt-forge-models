# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Infomaniak-AI vLLM TranslateGemma 27B IT model loader implementation for text translation.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
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


class ModelVariant(StrEnum):
    """Available Infomaniak-AI vLLM TranslateGemma model variants."""

    VLLM_TRANSLATEGEMMA_27B_IT = "vllm-translategemma-27b-it"


class ModelLoader(ForgeModel):
    """Infomaniak-AI vLLM TranslateGemma 27B IT model loader for text translation."""

    _VARIANTS = {
        ModelVariant.VLLM_TRANSLATEGEMMA_27B_IT: LLMModelConfig(
            pretrained_model_name="Infomaniak-AI/vllm-translategemma-27b-it",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VLLM_TRANSLATEGEMMA_27B_IT

    # The modified chat template encodes source/target language codes directly in
    # the content string using delimiters: <<<source>>>{src}<<<target>>>{tgt}<<<text>>>{text}
    sample_messages = [
        {
            "role": "user",
            "content": "<<<source>>>cs<<<target>>>de-DE<<<text>>>V nejhorším případě i k prasknutí čočky.",
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="vllm-translategemma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        inputs = self.processor.apply_chat_template(
            self.sample_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
