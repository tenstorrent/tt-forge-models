# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
inferencerlabs/Qwen3.5-27B-MLX-9bit model loader implementation for image to text.
"""

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


class ModelVariant(StrEnum):
    """Available inferencerlabs Qwen3.5-27B-MLX-9bit model variants for image to text."""

    QWEN_3_5_27B_MLX_9BIT = "27B_MLX_9bit"


class ModelLoader(ForgeModel):
    """inferencerlabs Qwen3.5-27B-MLX-9bit model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_MLX_9BIT: LLMModelConfig(
            pretrained_model_name="inferencerlabs/Qwen3.5-27B-MLX-9bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_MLX_9BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="inferencerlabs Qwen3.5-27B-MLX-9bit",
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

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        prompt = "Describe what you see in an image of a sunny beach."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs
