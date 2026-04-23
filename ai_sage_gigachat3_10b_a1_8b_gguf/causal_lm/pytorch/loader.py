# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ai-sage GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling.
"""
import importlib.util
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

_orig_is_gguf_available = _gguf_utils.is_gguf_available


def _patched_is_gguf_available(*args, **kwargs):
    if importlib.util.find_spec("gguf") is None:
        return False
    try:
        return _orig_is_gguf_available(*args, **kwargs)
    except Exception:
        return True


_gguf_utils.is_gguf_available = _patched_is_gguf_available


def _patch_deepseek2_gguf_support():
    """Register deepseek2 GGUF architecture and map it to HF deepseek_v2 model type."""
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    if "deepseek2" in GGUF_CONFIG_MAPPING:
        return

    # Map deepseek2 GGUF config keys to HF DeepseekV2Config fields.
    GGUF_CONFIG_MAPPING["deepseek2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "leading_dense_block_count": "first_k_dense_replace",
        "attention.kv_lora_rank": "kv_lora_rank",
        "attention.key_length_mla": "qk_rope_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "expert_shared_count": "n_shared_experts",
    }

    # GGUF_SUPPORTED_ARCHITECTURES was built from GGUF_CONFIG_MAPPING at import time;
    # append directly so load_gguf_checkpoint accepts deepseek2 files.
    if "deepseek2" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")

    # Add tokenizer converter for deepseek2 architecture key.
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek2", GGUFLlamaConverter)
    GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFLlamaConverter)

    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        # Remap deepseek2 → deepseek_v2 so AutoModelForCausalLM picks the right class.
        if result.get("config", {}).get("model_type") == "deepseek2":
            result["config"]["model_type"] = "deepseek_v2"
        return result

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        # gguf-py uses "deepseek2"; HF uses "deepseek_v2"
        if model_type == "deepseek_v2":
            model_type = "deepseek2"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_deepseek2_gguf_support()

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
    """Available ai-sage GigaChat3 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT3_10B_A1_8B_GGUF = "10B_A1_8B_GGUF"


class ModelLoader(ForgeModel):
    """ai-sage GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT3_10B_A1_8B_GGUF: LLMModelConfig(
            pretrained_model_name="ai-sage/GigaChat3-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT3_10B_A1_8B_GGUF

    GGUF_FILE = "GigaChat3-10B-A1.8B-q6_k.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="ai-sage GigaChat3 10B A1.8B GGUF",
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
