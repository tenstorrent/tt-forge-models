# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher GPT-OSS-Swallow 120B RL v0.1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_gpt_oss_support():
    """Register gpt-oss architecture as an alias for qwen3_moe.

    GPT-OSS 120B uses the same model architecture as Qwen3 MoE but the GGUF
    file declares architecture as 'gpt-oss' which transformers does not
    recognise.
    """
    if "gpt-oss" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("gpt-oss")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3_moe" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3_moe"]
            )
            mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            mapping["attention.sliding_window"] = "sliding_window"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["gpt-oss"] = mapping
    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["gpt-oss"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "qwen3_moe" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "gpt-oss"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3_moe"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add gpt-oss support and fix model_type."""
    _patch_gpt_oss_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "gpt-oss":
        result["config"]["model_type"] = "qwen3_moe"
    return result


_patch_gpt_oss_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available mradermacher GPT-OSS-Swallow 120B RL v0.1 GGUF model variants for causal language modeling."""

    GPT_OSS_SWALLOW_120B_RL_V0_1_Q4_K_M_GGUF = "120B_RL_V0_1_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher GPT-OSS-Swallow 120B RL v0.1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GPT_OSS_SWALLOW_120B_RL_V0_1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GPT-OSS-Swallow-120B-RL-v0.1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_SWALLOW_120B_RL_V0_1_Q4_K_M_GGUF

    GGUF_FILE = "GPT-OSS-Swallow-120B-RL-v0.1.Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None
