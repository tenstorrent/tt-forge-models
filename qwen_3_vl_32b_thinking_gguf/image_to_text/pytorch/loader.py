# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from typing import Optional

from ....base import ForgeModel


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture as an alias for qwen3, targeting qwen3_vl model type."""
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]
        if "qwen3" in mapping:
            mapping.setdefault("qwen3vl", mapping["qwen3"])


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl GGUF architecture support."""
    _patch_qwen3vl_gguf_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


_patch_qwen3vl_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
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
    """Available Qwen 3 VL 32B Thinking GGUF model variants for image to text."""

    QWEN_3_VL_32B_THINKING_1M_GGUF = "32b_thinking_1m_gguf"
    QWEN_3_VL_32B_THINKING_GGUF = "32b_thinking_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 32B Thinking GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-32B-Thinking-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Thinking-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_32B_THINKING_1M_GGUF: "Qwen3-VL-32B-Thinking-1M-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_32B_THINKING_GGUF: "Qwen3VL-32B-Thinking-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 32B Thinking GGUF",
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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Thinking")

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
