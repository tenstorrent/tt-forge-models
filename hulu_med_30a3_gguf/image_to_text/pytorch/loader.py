# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hulu-Med 30A3 GGUF model loader implementation for medical image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
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


def _patch_qwen3vlmoe_support():
    """Register qwen3vlmoe architecture as an alias for qwen3_moe in GGUF loading.

    The GGUF file for Qwen3-VL-MoE models declares architecture 'qwen3vlmoe',
    which transformers does not yet recognise in its GGUF loader. The text
    backbone has the same parameter layout as qwen3_moe, so we reuse that
    mapping and then fix up the model_type to qwen3_vl_moe after loading.
    """
    if "qwen3vlmoe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vlmoe",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"],
            )
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3vlmoe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    _patch_qwen3vlmoe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    if result.get("config", {}).get("model_type") == "qwen3vlmoe":
        result["config"]["model_type"] = "qwen3_vl_moe"
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kwargs):
    if model_type is None:
        model_type = hf_model.config.model_type
    # gguf-py uses 'qwen3vlmoe' but transformers uses 'qwen3_vl_moe'
    if model_type == "qwen3_vl_moe":
        model_type = "qwen3vlmoe"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type=model_type, **kwargs
    )


_patch_qwen3vlmoe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


class ModelVariant(StrEnum):
    """Available Hulu-Med 30A3 GGUF model variants for image to text."""

    HULU_MED_30A3_GGUF = "30a3_gguf"


class ModelLoader(ForgeModel):
    """Hulu-Med 30A3 GGUF model loader implementation for medical image to text tasks."""

    _VARIANTS = {
        ModelVariant.HULU_MED_30A3_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Hulu-Med-30A3-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HULU_MED_30A3_GGUF

    GGUF_FILE = "Hulu-Med-30A3.Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hulu-Med 30A3 GGUF",
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
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "ZJU-AI4H/Hulu-Med-30A3", trust_remote_code=True
        )

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
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
                    {
                        "type": "text",
                        "text": "Generate a medical report for this image.",
                    },
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
