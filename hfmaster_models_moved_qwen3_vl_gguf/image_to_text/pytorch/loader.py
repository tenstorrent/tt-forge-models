# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
hfmaster models-moved Qwen3-VL GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_base as _tok_utils
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_qwen3vl_support():
    """Register qwen3vl architecture as an alias for qwen3_vl.

    Transformers 5.x has Qwen3VLForConditionalGeneration but lacks GGUF
    loading support for the qwen3vl architecture declared in GGUF files.
    """
    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    qwen3_config = GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3", {})
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"] = dict(qwen3_config)
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vl"].update(
        {
            "attention.key_length": "head_dim",
            "attention.value_length": None,
            "rope.dimension_sections": None,
            "n_deepstack_layers": None,
        }
    )
    for section in ("tokenizer", "tokenizer_config"):
        qwen3_tok = GGUF_TO_TRANSFORMERS_MAPPING[section].get("qwen3", {})
        GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault("qwen3vl", dict(qwen3_tok))
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])
    elif "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUFQwen2Converter)


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and fix model_type."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "qwen3vl":
        result["config"]["model_type"] = "qwen3_vl"
    return result


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available hfmaster models-moved Qwen3-VL GGUF variants for image to text."""

    QWEN3_VL_8B_INSTRUCT_ABLITERATED_V2_Q4_K_M_GGUF = (
        "8b_instruct_abliterated_v2_q4_k_m_gguf"
    )


class ModelLoader(ForgeModel):
    """hfmaster models-moved Qwen3-VL GGUF model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_VL_8B_INSTRUCT_ABLITERATED_V2_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="hfmaster/models-moved",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_VL_8B_INSTRUCT_ABLITERATED_V2_Q4_K_M_GGUF

    GGUF_FILE = "qwen3vl/Qwen3-VL-8B-Instruct-abliterated-v2.0.Q4_K_M.gguf"
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="hfmaster models-moved Qwen3-VL GGUF",
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

        # Use base model config so vision_config and special token IDs are correct;
        # hfmaster/models-moved does not carry a Qwen3-VL config.json.
        config = AutoConfig.from_pretrained(self.BASE_MODEL)
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

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
