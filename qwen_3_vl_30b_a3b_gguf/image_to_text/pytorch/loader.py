# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL 30B A3B GGUF model loader implementation for image to text.
"""

import re

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import (
    AutoProcessor,
    Qwen3VLMoeForConditionalGeneration,
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

# Keys that stay at the top level of the VL config (not moved to text_config)
_VL_TOP_LEVEL_KEYS = {"model_type", "architectures"}


class _Qwen3VLMoeTensorProcessor(_gguf_utils.Qwen2MoeTensorProcessor):
    """Extends Qwen2MoeTensorProcessor for Qwen3-VL-MoE weight naming.

    Qwen3-VL-MoE stores language model weights under model.language_model.*
    so during submodule recursion the names lack the model. prefix. We extend
    the fallback patterns to cover both forms.
    """

    # Match layers.N.* (VL submodule) and model.layers.N.* (non-VL top-level)
    HF_MOE_W13_PATTERN = re.compile(
        r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj"
    )
    _HF_MOE_DOWN_PATTERN = re.compile(
        r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.experts\.down_proj"
    )
    _HF_MOE_GATE_INP_PATTERN = re.compile(
        r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.gate"
    )

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        if m := re.fullmatch(self.HF_MOE_W13_PATTERN, hf_name):
            full_hf_name = qual_name + hf_name
            bid = m["bid"]
            gguf_to_hf_name_map[f"blk.{bid}.ffn_gate_exps{suffix}"] = full_hf_name
            gguf_to_hf_name_map[f"blk.{bid}.ffn_up_exps{suffix}"] = full_hf_name
        elif m := re.fullmatch(self._HF_MOE_DOWN_PATTERN, hf_name):
            full_hf_name = qual_name + hf_name
            bid = m["bid"]
            gguf_to_hf_name_map[f"blk.{bid}.ffn_down_exps{suffix}"] = full_hf_name
        elif m := re.fullmatch(self._HF_MOE_GATE_INP_PATTERN, hf_name):
            full_hf_name = qual_name + hf_name
            bid = m["bid"]
            gguf_to_hf_name_map[f"blk.{bid}.ffn_gate_inp{suffix}"] = full_hf_name


def _patch_qwen3vlmoe_gguf_support():
    """Register qwen3vlmoe GGUF architecture for Qwen3-VL-30B-A3B (MoE).

    Transformers 5.x has Qwen3VLMoeForConditionalGeneration but lacks GGUF
    loading support for qwen3vlmoe. We bridge the gap by:
    - Registering qwen3vlmoe in GGUF_SUPPORTED_ARCHITECTURES
    - Adding config/tensor mappings (reuse qwen3_moe)
    - Transforming the flat GGUF config into the nested text_config structure
      required by Qwen3VLMoeConfig
    - Patching get_gguf_hf_weights_map to map qwen3_vl_moe -> qwen3vlmoe,
      source num_hidden_layers from text_config, and use the VL-aware processor
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3vlmoe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vlmoe")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3vlmoe"] = dict(
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3_moe"]
    )

    TENSOR_PROCESSORS["qwen3vlmoe"] = _Qwen3VLMoeTensorProcessor

    for alias in ("qwen3vlmoe", "qwen3_vl_moe", "qwen3_vl_moe_text"):
        if (
            alias not in GGUF_TO_FAST_CONVERTERS
            and "qwen3_moe" in GGUF_TO_FAST_CONVERTERS
        ):
            GGUF_TO_FAST_CONVERTERS[alias] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]

    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3vlmoe":
            text_cfg = {"model_type": "qwen3_vl_moe_text"}
            top_cfg = {"model_type": "qwen3_vl_moe"}
            for k, v in config.items():
                if k in _VL_TOP_LEVEL_KEYS:
                    if k != "model_type":
                        top_cfg[k] = v
                else:
                    text_cfg[k] = v
            top_cfg["text_config"] = text_cfg
            # out_hidden_size must match the text model's hidden_size (default 3584 is wrong for this model)
            top_cfg["vision_config"] = {
                "out_hidden_size": text_cfg.get("hidden_size", 3584)
            }
            result["config"] = top_cfg
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_vl_moe", "qwen3_vl_moe_text"):
            model_type = "qwen3vlmoe"
            if not isinstance(processor, _Qwen3VLMoeTensorProcessor):
                processor = _Qwen3VLMoeTensorProcessor()
        if num_layers is None:
            cfg = hf_model.config
            if hasattr(cfg, "num_hidden_layers"):
                num_layers = cfg.num_hidden_layers
            elif hasattr(cfg, "text_config") and hasattr(
                cfg.text_config, "num_hidden_layers"
            ):
                num_layers = cfg.text_config.num_hidden_layers
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen3vlmoe_gguf_support()


class ModelVariant(StrEnum):
    """Available Qwen 3 VL 30B A3B GGUF model variants for image to text."""

    QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF = "30b_a3b_instruct_1m_gguf"
    QWEN_3_VL_30B_A3B_INSTRUCT_GGUF = "30b_a3b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL 30B A3B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-VL-30B-A3B-Instruct-1M-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_1M_GGUF: "Qwen3-VL-30B-A3B-Instruct-1M-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_30B_A3B_INSTRUCT_GGUF: "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL 30B A3B GGUF",
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
        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs["ignore_mismatched_sizes"] = True
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

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
