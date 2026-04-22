# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Samantha Big MoE i1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, Llama4ForCausalLM, Llama4TextConfig
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


def _patch_gguf_llama4_support():
    """Patch transformers' GGUF loading to support llama4 architecture."""
    import transformers.integrations.ggml as ggml_mod
    import transformers.modeling_gguf_pytorch_utils as gguf_utils_mod
    from transformers.integrations.ggml import GGUFLlamaConverter

    if "llama4" in ggml_mod.GGUF_CONFIG_MAPPING:
        return

    ggml_mod.GGUF_CONFIG_MAPPING["llama4"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size_mlp",
        "expert_feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "interleave_moe_layer_step": "interleave_moe_layer_step",
    }

    if "llama4" not in gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES:
        gguf_utils_mod.GGUF_SUPPORTED_ARCHITECTURES.append("llama4")

    if "llama4" not in ggml_mod.GGUF_TO_FAST_CONVERTERS:
        ggml_mod.GGUF_TO_FAST_CONVERTERS["llama4"] = GGUFLlamaConverter


def _load_llama4_text_config_from_gguf(gguf_path):
    """Build Llama4TextConfig by reading GGUF metadata directly."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    fields = {}
    for name, field in reader.fields.items():
        if field.data and len(field.data) == 1:
            val = field.parts[field.data[0]]
            if hasattr(val, "tolist"):
                val = val.tolist()
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
            fields[name] = val

    return Llama4TextConfig(
        vocab_size=fields.get("llama4.vocab_size", 202048),
        hidden_size=fields.get("llama4.embedding_length", 5120),
        intermediate_size=fields.get("llama4.expert_feed_forward_length", 8192),
        intermediate_size_mlp=fields.get("llama4.feed_forward_length", 16384),
        num_hidden_layers=fields.get("llama4.block_count", 48),
        num_attention_heads=fields.get("llama4.attention.head_count", 40),
        num_key_value_heads=fields.get("llama4.attention.head_count_kv", 8),
        head_dim=fields.get("llama4.attention.key_length", 128),
        rms_norm_eps=fields.get("llama4.attention.layer_norm_rms_epsilon", 1e-5),
        num_local_experts=fields.get("llama4.expert_count", 16),
        num_experts_per_tok=fields.get("llama4.expert_used_count", 1),
        interleave_moe_layer_step=fields.get("llama4.interleave_moe_layer_step", 1),
    )


class ModelVariant(StrEnum):
    """Available Samantha Big MoE i1 GGUF model variants for causal language modeling."""

    SAMANTHA_BIG_MOE_I1_IQ3_M_GGUF = "i1-IQ3_M"


class ModelLoader(ForgeModel):
    """Samantha Big MoE i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SAMANTHA_BIG_MOE_I1_IQ3_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Samantha-big-MoE-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAMANTHA_BIG_MOE_I1_IQ3_M_GGUF

    GGUF_FILE = "Samantha-big-MoE.i1-IQ3_M.gguf"

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
            model="Samantha Big MoE i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_gguf_llama4_support()

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

    def _get_gguf_path(self):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=self.GGUF_FILE,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _patch_gguf_llama4_support()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        gguf_path = self._get_gguf_path()
        text_config = _load_llama4_text_config_from_gguf(gguf_path)

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers

        model = Llama4ForCausalLM(text_config).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

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
        _patch_gguf_llama4_support()
        gguf_path = self._get_gguf_path()
        self.config = _load_llama4_text_config_from_gguf(gguf_path)
        return self.config
