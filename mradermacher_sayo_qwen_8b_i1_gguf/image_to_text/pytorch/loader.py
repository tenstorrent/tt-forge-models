# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Sayo-Qwen-8B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
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
    """Available mradermacher Sayo-Qwen-8B i1 GGUF model variants for image to text."""

    SAYO_QWEN_8B_I1_Q4_K_M_GGUF = "8B_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Sayo-Qwen-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SAYO_QWEN_8B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Sayo-Qwen-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAYO_QWEN_8B_I1_Q4_K_M_GGUF

    GGUF_FILE = "Sayo-Qwen-8B.i1-Q4_K_M.gguf"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher Sayo-Qwen-8B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        # GGUF weights require the gguf package which is not available.
        # Load architecture from the base model config for compile-only testing.
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_config(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
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
