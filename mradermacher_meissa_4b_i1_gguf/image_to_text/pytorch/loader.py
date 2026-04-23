# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/Meissa-4B-i1-GGUF model loader implementation for image to text.
"""

from transformers import (
    Qwen3VLConfig,
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
    """Available mradermacher Meissa-4B-i1-GGUF model variants for image to text."""

    MEISSA_4B_I1_GGUF = "4b_i1_gguf"


class ModelLoader(ForgeModel):
    """mradermacher Meissa-4B-i1-GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEISSA_4B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Meissa-4B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEISSA_4B_I1_GGUF

    GGUF_FILE = "Meissa-4B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mradermacher Meissa-4B-i1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import importlib.metadata

        import transformers.utils.import_utils as _transformers_import_utils

        # gguf is installed at runtime; refresh the static mapping so
        # transformers' is_gguf_available() can look up the version.
        if "gguf" not in _transformers_import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _transformers_import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = [
                    "gguf"
                ]
            except importlib.metadata.PackageNotFoundError:
                pass

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        # Pass explicit config so GGUF loading skips the architecture
        # check (qwen3vl is not yet registered in transformers GGUF support).
        config = Qwen3VLConfig()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
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
