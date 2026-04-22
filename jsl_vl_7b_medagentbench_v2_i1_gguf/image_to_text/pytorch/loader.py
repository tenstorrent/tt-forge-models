# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF model loader implementation for image to text.

Note: The qwen2vl GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the HF-native Qwen2.5-VL-7B-Instruct checkpoint.
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
    """Available Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF variants for image to text."""

    JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF = "7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF loader for image to text tasks.

    Note: Uses Qwen/Qwen2.5-VL-7B-Instruct (safetensors) instead of GGUF because
    the qwen2vl GGUF architecture is not yet supported by transformers.
    """

    _VARIANTS = {
        ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JSL_VL_7B_MEDAGENTBENCH_V2_I1_Q4_K_M_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mradermacher JSL-VL-7B-MedAgentBench-v2 i1-GGUF",
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
            pretrained_model_name,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
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
