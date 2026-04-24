# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 Scout GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Llama4ForCausalLM,
)
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


def _patch_llama4_gguf_support():
    """Register llama4 GGUF architecture support in transformers.

    Transformers 5.x has Llama4ForCausalLM but lacks GGUF loading support for
    the llama4 architecture. We register the config mapping, tokenizer
    converter, and post-load fixup to bridge the gap.
    """
    if "llama4" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("llama4")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["llama4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size_mlp",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "intermediate_size",
    }

    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["llama4"] = GGUF_TO_FAST_CONVERTERS["llama"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_llama4_gguf_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    return result


_patch_llama4_gguf_support()
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
    """Available Llama 4 Scout GGUF model variants for causal language modeling."""

    LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF = "17B_16E_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """Llama 4 Scout GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT_GGUF

    GGUF_FILE = "Llama-4-Scout-17B-16E-Instruct-Q3_K_S.gguf"

    sample_text = "What are the main advantages of mixture-of-experts models?"

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
            model="Llama 4 Scout GGUF",
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

        config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        text_config = config.get_text_config()

        num_layers = self.num_layers if self.num_layers is not None else 6
        text_config.num_hidden_layers = num_layers
        text_config.num_attention_heads = 16
        text_config.hidden_size = 1024
        text_config.num_key_value_heads = 16
        text_config.intermediate_size = 1024 * 4
        text_config.intermediate_size_mlp = 1024 * 4
        text_config.num_local_experts = 16
        text_config.num_experts_per_tok = 1

        model_kwargs = {"attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Llama4ForCausalLM._from_config(text_config, **model_kwargs)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
