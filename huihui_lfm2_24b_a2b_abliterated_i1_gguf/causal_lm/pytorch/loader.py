# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    TENSOR_PROCESSORS,
    Lfm2TensorProcessor,
)

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


def _patch_lfm2moe_support():
    """Register lfm2moe GGUF architecture as lfm2_moe in transformers.

    LFM2 24B A2B uses GGUF architecture 'lfm2moe' (no underscore) but
    transformers only registers 'lfm2' (dense variant). The HF model_type
    is 'lfm2_moe'. Extend the mappings to cover the MoE variant.
    """
    if "lfm2moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    lfm2moe_config_map = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "vocab_size": "vocab_size",
        "shortconv.l_cache": "conv_L_cache",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "leading_dense_block_count": "num_dense_layers",
    }
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["lfm2moe"] = lfm2moe_config_map
    GGUF_SUPPORTED_ARCHITECTURES.append("lfm2moe")
    TENSOR_PROCESSORS["lfm2moe"] = Lfm2TensorProcessor


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add lfm2moe support and fix model_type."""
    _patch_lfm2moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "lfm2moe":
        result["config"]["model_type"] = "lfm2_moe"
        # GGUF stores num_key_value_heads as a per-layer list (same as lfm2 dense).
        # 0 means short-conv layer (no attention); non-zero is the KV head count.
        gguf_num_kv_heads = result["config"].get("num_key_value_heads", [])
        if isinstance(gguf_num_kv_heads, list):
            result["config"]["num_key_value_heads"] = (
                max(gguf_num_kv_heads) if gguf_num_kv_heads else 0
            )
            result["config"]["full_attn_idxs"] = [
                i for i, n in enumerate(gguf_num_kv_heads) if n > 0
            ]
        elif gguf_num_kv_heads == 0:
            result["config"]["num_key_value_heads"] = result["config"].get(
                "num_attention_heads", 32
            )
    return result


_patch_lfm2moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Huihui LFM2 24B A2B Abliterated i1-GGUF model variants for causal language modeling."""

    HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF = "24B_A2B_i1_GGUF"


class ModelLoader(ForgeModel):
    """Huihui LFM2 24B A2B Abliterated i1-GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-LFM2-24B-A2B-abliterated-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_LFM2_24B_A2B_ABLITERATED_I1_GGUF

    GGUF_FILE = "Huihui-LFM2-24B-A2B-abliterated.i1-Q4_K_M.gguf"

    sample_text = "The quick brown fox jumps over the lazy dog."

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
            model="Huihui LFM2 24B A2B Abliterated i1-GGUF",
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
