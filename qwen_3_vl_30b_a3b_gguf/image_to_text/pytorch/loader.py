# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 30B A3B GGUF model loader implementation for image to text.
"""

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)
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
    """Available Qwen 3 VL 30B A3B GGUF model variants for image to text."""

    QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF = "30b_a3b_instruct_1m_gguf"
    QWEN_3_VL_30B_A3B_INSTRUCT_GGUF = "30b_a3b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 30B A3B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 30B A3B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

        # The transformers GGUF loader does not yet support the qwen3vlmoe architecture,
        # so we load the config from the base model and instantiate with random weights.
        # For compile-only environments this is acceptable.
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

        target_dtype = dtype_override if dtype_override is not None else torch.float32
        old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            model = AutoModelForImageTextToText.from_config(config)
        finally:
            torch.set_default_dtype(old_default_dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
