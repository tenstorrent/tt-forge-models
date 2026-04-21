# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Sayo Qwen 8B GGUF model loader implementation for image to text.
"""

import importlib.metadata
from typing import Optional

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

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
    """Available mradermacher Sayo Qwen 8B GGUF model variants for image to text."""

    SAYO_QWEN_8B_GGUF = "8b_gguf"


class ModelLoader(ForgeModel):
    """mradermacher Sayo Qwen 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.SAYO_QWEN_8B_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Sayo-Qwen-8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAYO_QWEN_8B_GGUF

    GGUF_FILE = "Sayo-Qwen-8B.Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @staticmethod
    def _refresh_gguf_package_mapping():
        # transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time; gguf
        # is installed later by RequirementsManager so the cache misses it,
        # causing version.parse('N/A') to raise InvalidVersion in is_gguf_available().
        try:
            import transformers.utils.import_utils as _tfu

            if "gguf" not in _tfu.PACKAGE_DISTRIBUTION_MAPPING:
                _tfu.PACKAGE_DISTRIBUTION_MAPPING = (
                    importlib.metadata.packages_distributions()
                )
        except Exception:
            pass

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher Sayo Qwen 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._refresh_gguf_package_mapping()
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
