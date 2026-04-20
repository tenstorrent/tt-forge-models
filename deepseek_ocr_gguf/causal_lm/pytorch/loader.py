# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR GGUF model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import AutoTokenizer
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


def _load_deepseek_v2_config_from_gguf(gguf_path):
    """Build a DeepseekV2Config directly from GGUF metadata.

    The DeepSeek-OCR GGUF uses 'deepseek_vl_v2' architecture with bare
    (non-prefixed) config keys that transformers' GGUF loader cannot parse.
    We read them directly and construct the config ourselves.
    """
    from gguf import GGUFReader
    from transformers.modeling_gguf_pytorch_utils import _gguf_parse_value
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
        DeepseekV2Config,
    )

    reader = GGUFReader(gguf_path)
    raw = {}
    for key, field in reader.fields.items():
        try:
            raw[key] = _gguf_parse_value(field.parts[field.data[0]], field.types)
        except Exception:
            pass

    return DeepseekV2Config(
        vocab_size=raw.get("vocab_size", 129280),
        hidden_size=raw.get("hidden_size", 1280),
        intermediate_size=raw.get("intermediate_size", 6848),
        moe_intermediate_size=raw.get("moe_intermediate_size", 896),
        num_hidden_layers=raw.get("num_hidden_layers", 12),
        num_attention_heads=raw.get("num_attention_heads", 10),
        num_key_value_heads=raw.get("num_key_value_heads", 10),
        hidden_act=raw.get("hidden_act", "silu"),
        max_position_embeddings=raw.get("max_position_embeddings", 8192),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        rope_parameters={"rope_type": "default"},
        attention_bias=raw.get("attention_bias", False),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        n_shared_experts=raw.get("n_shared_experts", 2),
        n_routed_experts=raw.get("n_routed_experts", 64),
        num_experts_per_tok=raw.get("num_experts_per_tok", 6),
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.0),
        topk_method=raw.get("topk_method", "greedy"),
        n_group=raw.get("n_group", 1),
        topk_group=raw.get("topk_group", 1),
        first_k_dense_replace=raw.get("first_k_dense_replace", 1),
        norm_topk_prob=raw.get("norm_topk_prob", False),
        bos_token_id=raw.get("bos_token_id", 0),
        eos_token_id=raw.get("eos_token_id", 1),
    )


class ModelVariant(StrEnum):
    """Available DeepSeek OCR GGUF model variants for causal language modeling."""

    DEEPSEEK_OCR_Q4_0 = "Q4_0"


class ModelLoader(ForgeModel):
    """DeepSeek OCR GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_Q4_0: LLMModelConfig(
            pretrained_model_name="NexaAI/DeepSeek-OCR-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_Q4_0

    GGUF_FILE = "DeepSeek-OCR.Q4_0.gguf"

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
            model="DeepSeek OCR GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        config = self.load_config()
        if self.tokenizer.eos_token is None and config.eos_token_id is not None:
            self.tokenizer.eos_token = self.tokenizer.convert_ids_to_tokens(
                config.eos_token_id
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import DeepseekV2ForCausalLM

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self.load_config()
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = DeepseekV2ForCausalLM(config)
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        if self.config is not None:
            return self.config
        from huggingface_hub import hf_hub_download

        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name, self.GGUF_FILE
        )
        self.config = _load_deepseek_v2_config_from_gguf(gguf_path)
        return self.config
