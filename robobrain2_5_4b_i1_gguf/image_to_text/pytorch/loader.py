# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoboBrain2.5-4B i1 GGUF model loader implementation for image to text.
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

_QWEN3VL_TEXT_CONFIG_KEYS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
}


def _patch_qwen3vl_support():
    """Register qwen3vl GGUF architecture as an alias for qwen3_vl (Qwen3 VL).

    Transformers uses model_type 'qwen3_vl' but GGUF files declare
    architecture as 'qwen3vl'. We register the mapping and restructure the
    flat GGUF config into the nested text_config/vision_config that
    Qwen3VLConfig expects.
    """
    if "qwen3vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen3vl",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3vl", GGUF_TO_FAST_CONVERTERS["qwen3"])


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3vl support and restructure config."""
    _patch_qwen3vl_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    config = result.get("config", {})
    if config.get("model_type") == "qwen3vl":
        config["model_type"] = "qwen3_vl"
        text_cfg = {k: config.pop(k) for k in _QWEN3VL_TEXT_CONFIG_KEYS if k in config}
        if text_cfg:
            config["text_config"] = text_cfg
    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor=None, model_type=None, num_layers=None, qual_name=""
):
    """Wrap get_gguf_hf_weights_map to handle Qwen3VL composite config.

    Qwen3VLConfig stores num_hidden_layers in text_config, and the gguf-py
    arch name is 'qwen3vl' while transformers model_type is 'qwen3_vl'.
    """
    cfg = hf_model.config
    if model_type is None and getattr(cfg, "model_type", None) == "qwen3_vl":
        model_type = "qwen3vl"
    if num_layers is None and getattr(cfg, "model_type", None) == "qwen3_vl":
        num_layers = cfg.text_config.num_hidden_layers
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_qwen3vl_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

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
    """Available RoboBrain2.5-4B i1 GGUF model variants for image to text."""

    ROBOBRAIN2_5_4B_I1_Q4_K_M_GGUF = "2_5_4b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """RoboBrain2.5-4B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBOBRAIN2_5_4B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/RoboBrain2.5-4B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBOBRAIN2_5_4B_I1_Q4_K_M_GGUF

    GGUF_FILE = "RoboBrain2.5-4B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RoboBrain2.5-4B i1 GGUF",
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
        self.processor = AutoProcessor.from_pretrained("BAAI/RoboBrain2.5-4B")

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
