# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3 VL 4B Abliterated GGUF model loader implementation for image to text.
"""

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING, GGUF_TO_FAST_CONVERTERS

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

_TEXT_CONFIG_KEYS = {
    "max_position_embeddings",
    "num_hidden_layers",
    "intermediate_size",
    "hidden_size",
    "rope_theta",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "vocab_size",
}


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture as an alias for qwen3_vl.

    llama.cpp serialises Qwen3-VL models with general.architecture = "qwen3vl"
    but transformers does not recognise this name. Map it through the existing
    qwen3 config key table and post-process the flat config dict into the nested
    text_config / vision_config structure expected by Qwen3VLConfig.
    """
    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_CONFIG_MAPPING["qwen3vl"] = dict(GGUF_CONFIG_MAPPING.get("qwen3", {}))
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3vl"] = GGUF_TO_FAST_CONVERTERS["qwen3"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support."""
    _patch_qwen3vl_gguf_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    cfg = result.get("config", {})
    if cfg.get("model_type") != "qwen3vl":
        return result
    # Restructure flat text-config keys into the nested dict Qwen3VLConfig expects.
    text_cfg = {k: cfg.pop(k) for k in list(cfg) if k in _TEXT_CONFIG_KEYS}
    if text_cfg:
        cfg["text_config"] = text_cfg
    cfg["model_type"] = "qwen3_vl"
    return result


_patch_qwen3vl_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Huihui Qwen3 VL 4B Abliterated GGUF model variants for image to text."""

    HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF = "4b_instruct_abliterated_gguf"
    HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF = (
        "4b_instruct_abliterated_mradermacher_gguf"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen3 VL 4B Abliterated GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_GGUF: "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf",
        ModelVariant.HUIHUI_QWEN3_VL_4B_INSTRUCT_ABLITERATED_MRADERMACHER_GGUF: "Huihui-Qwen3-VL-4B-Instruct-abliterated.Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Huihui Qwen3 VL 4B Abliterated GGUF",
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
