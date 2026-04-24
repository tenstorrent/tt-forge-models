# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GELab-Zero-4B-preview GGUF model loader implementation for image to text.
"""

import importlib.metadata

import transformers.utils.import_utils as _import_utils

# gguf has no __version__ attribute; transformers captures PACKAGE_DISTRIBUTION_MAPPING
# at import time (before requirements.txt installs gguf), so version lookup falls through
# to getattr(gguf, '__version__', 'N/A') which returns 'N/A', then version.parse('N/A')
# raises InvalidVersion. Refreshing the mapping and clearing the lru_cache fixes this.
_import_utils.PACKAGE_DISTRIBUTION_MAPPING.update(
    importlib.metadata.packages_distributions()
)
_import_utils.is_gguf_available.cache_clear()

from transformers import (
    Qwen3VLForConditionalGeneration,
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
    """Available GELab-Zero-4B-preview GGUF model variants for image to text."""

    GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF = "4B_PREVIEW_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """GELab-Zero-4B-preview GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Mungert/GELab-Zero-4B-preview-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GELAB_ZERO_4B_PREVIEW_Q4_K_M_GGUF

    GGUF_FILE = "GELab-Zero-4B-preview-q4_k_m.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GELab-Zero-4B-preview GGUF",
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

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

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
