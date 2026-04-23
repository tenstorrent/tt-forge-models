# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wizardeur Qwen3.5-9B GPTQ-marlin model loader implementation for image to text.
"""

from gptqmodel import GPTQModel, BACKEND
from transformers import (
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
    """Available Wizardeur Qwen3.5-9B GPTQ-marlin model variants for image to text."""

    QWEN3_5_9B_GPTQ_MARLIN = "9B_GPTQ_marlin"


class ModelLoader(ForgeModel):
    """Wizardeur Qwen3.5-9B GPTQ-marlin model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_9B_GPTQ_MARLIN: LLMModelConfig(
            pretrained_model_name="wizardeur/Qwen3.5-9B-GPTQ-marlin",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_9B_GPTQ_MARLIN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wizardeur Qwen3.5-9B GPTQ-marlin",
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
        model_kwargs.pop("device_map", None)

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = GPTQModel.load(
            pretrained_model_name, backend=BACKEND.TORCH, device="cpu", **model_kwargs
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
