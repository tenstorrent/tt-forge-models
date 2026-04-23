# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/translategemma-4b-it-4bit model loader implementation for text translation.

A 4-bit MLX-quantized variant of google/translategemma-4b-it, a Gemma3
conditional generation model fine-tuned for multilingual translation.
"""
import torch
from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration
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
    """Available mlx-community/translategemma-4b-it-4bit model variants."""

    TRANSLATEGEMMA_4B_IT_4BIT = "translategemma-4b-it-4bit"


class ModelLoader(ForgeModel):
    """mlx-community/translategemma-4b-it-4bit model loader for text translation."""

    _VARIANTS = {
        ModelVariant.TRANSLATEGEMMA_4B_IT_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/translategemma-4b-it-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSLATEGEMMA_4B_IT_4BIT

    sample_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "cs",
                    "target_lang_code": "de-DE",
                    "text": "V nejhorším případě i k prasknutí čočky.",
                }
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mlx-community translategemma-4b-it-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        self.processor = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"ignore_mismatched_sizes": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # MLX quantization format is incompatible with transformers; remove the
        # attribute entirely so hasattr() returns False and loading proceeds
        # without quantization.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )
        model.eval()
        self.model = model
        self.config = model.config
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
