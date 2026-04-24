# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth LFM2 8B A1B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    Lfm2TensorProcessor,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

_LFM2MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "shortconv.l_cache": "conv_L_cache",
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "leading_dense_block_count": "num_dense_layers",
}


def _patch_lfm2moe_support():
    """Register lfm2moe GGUF architecture for the lfm2_moe transformers model."""
    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "lfm2moe"
    ] = _LFM2MOE_CONFIG_MAPPING
    _gguf_utils.TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor
    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["lfm2moe"] = GGUF_TO_FAST_CONVERTERS["gpt2"]
        GGUF_TO_FAST_CONVERTERS["lfm2_moe"] = GGUF_TO_FAST_CONVERTERS["gpt2"]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to add lfm2moe architecture support."""
    _patch_lfm2moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    if result.get("config", {}).get("model_type") == "lfm2moe":
        gguf_num_kv_heads = result["config"].get("num_key_value_heads")
        if isinstance(gguf_num_kv_heads, list):
            result["config"]["num_key_value_heads"] = max(gguf_num_kv_heads)
            result["config"]["full_attn_idxs"] = [
                i for i, n in enumerate(gguf_num_kv_heads) if n > 0
            ]
        result["config"]["block_auto_adjust_ff_dim"] = False
        result["config"]["model_type"] = "lfm2_moe"
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
    """Available Unsloth LFM2 8B A1B GGUF model variants for causal language modeling."""

    UNSLOTH_LFM2_8B_A1B_GGUF = "8B_A1B_GGUF"


class ModelLoader(ForgeModel):
    """Unsloth LFM2 8B A1B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.UNSLOTH_LFM2_8B_A1B_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/LFM2-8B-A1B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_LFM2_8B_A1B_GGUF

    GGUF_FILE = "LFM2-8B-A1B-Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Unsloth LFM2 8B A1B GGUF",
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
