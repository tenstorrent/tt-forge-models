# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher AdaReasoner 7B GGUF model loader implementation for image to text.

Note: The qwen2vl GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the HF-native checkpoint (AdaReasoner/AdaReasoner-7B-Randomized)
instead.
"""

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
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
    """Available Mradermacher AdaReasoner 7B GGUF variants for image to text."""

    ADAREASONER_7B_Q4_K_M_GGUF = "7B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher AdaReasoner 7B GGUF loader for image to text tasks.

    Note: Uses the base model (safetensors) instead of GGUF because the
    qwen2vl GGUF architecture is not yet supported by transformers.
    """

    _VARIANTS = {
        ModelVariant.ADAREASONER_7B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="AdaReasoner/AdaReasoner-7B-Randomized",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ADAREASONER_7B_Q4_K_M_GGUF

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher AdaReasoner 7B GGUF",
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

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )

        # Load from base safetensors model because qwen2vl GGUF architecture
        # is not yet supported by transformers.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
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
