# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LFM2 GGUF model loader implementation for causal language modeling.

Supports LiquidAI's LFM2 Mixture-of-Experts models in GGUF format.
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


def _patch_lfm2moe_support():
    """Register lfm2moe architecture as an alias for lfm2.

    LFM2-24B-A2B uses the MoE variant of the LFM2 architecture but the GGUF
    file declares architecture as 'lfm2moe' which transformers does not
    recognise.  The underlying model maps to transformers' lfm2_moe model type.
    """
    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "lfm2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            mapping = dict(_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["lfm2"])
            if section == "config":
                mapping["expert_count"] = "num_experts"
                mapping["expert_used_count"] = "num_experts_per_tok"
                mapping["expert_feed_forward_length"] = "moe_intermediate_size"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["lfm2moe"] = mapping
    if hasattr(_gguf_utils, "TENSOR_PROCESSORS"):
        if "lfm2" in _gguf_utils.TENSOR_PROCESSORS:
            _gguf_utils.TENSOR_PROCESSORS["lfm2moe"] = _gguf_utils.TENSOR_PROCESSORS[
                "lfm2"
            ]
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        if "lfm2" in _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING:
            _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING[
                "lfm2moe"
            ] = _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["lfm2"]
    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = GGUF_TO_FAST_CONVERTERS["llama"]


LFM2_MOE_LAYER_TYPES = [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "conv",
    "full_attention",
    "conv",
]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_lfm2moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "lfm2moe":
        result["config"]["model_type"] = "lfm2_moe"
    if result.get("config", {}).get("model_type") in ("lfm2_moe", "lfm2moe"):
        if not result["config"].get("layer_types"):
            result["config"]["layer_types"] = LFM2_MOE_LAYER_TYPES
        kv = result["config"].get("num_key_value_heads")
        if isinstance(kv, list):
            result["config"]["num_key_value_heads"] = max(kv)
    return result


_patch_lfm2moe_support()
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
    """Available LFM2 GGUF model variants for causal language modeling."""

    LFM2_24B_A2B_GGUF = "LFM2_24B_A2B_GGUF"


class ModelLoader(ForgeModel):
    """LFM2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LFM2_24B_A2B_GGUF: LLMModelConfig(
            pretrained_model_name="LiquidAI/LFM2-24B-A2B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LFM2_24B_A2B_GGUF

    GGUF_FILE = "LFM2-24B-A2B-Q4_K_M.gguf"

    sample_text = (
        "What are the key differences between classical and quantum computing?"
    )

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LFM2 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
