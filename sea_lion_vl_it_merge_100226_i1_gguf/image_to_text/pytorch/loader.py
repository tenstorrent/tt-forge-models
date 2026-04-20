# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text.
"""

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
    """Available mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model variants for image to text."""

    SEA_LION_VL_IT_MERGE_100226_I1_GGUF = "SEA_LION_VL_IT_Merge_100226_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/SEA-LION-VL-IT-Merge-100226-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEA_LION_VL_IT_MERGE_100226_I1_GGUF

    GGUF_FILE = "SEA-LION-VL-IT-Merge-100226.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher SEA-LION-VL-IT-Merge-100226 i1 GGUF",
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
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repo does not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "SEACrowd/SEA-LION-VL-IT-Merge-100226"
        )

        model = AutoModelForImageTextToText.from_pretrained(
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
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
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
