# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EditScore-Qwen3-VL-8B-Instruct model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel
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
    """Available EditScore-Qwen3-VL-8B-Instruct model variants for image to text."""

    EDITSCORE_QWEN3_VL_8B_INSTRUCT = "editscore_qwen3_vl_8b_instruct"


class ModelLoader(ForgeModel):
    """EditScore-Qwen3-VL-8B-Instruct model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.EDITSCORE_QWEN3_VL_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="EditScore/EditScore-Qwen3-VL-8B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EDITSCORE_QWEN3_VL_8B_INSTRUCT

    BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="editscore_qwen3_vl_8b_instruct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        adapter_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["dtype"] = "auto"
        model_kwargs["device_map"] = "auto"
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL_NAME)

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()
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
