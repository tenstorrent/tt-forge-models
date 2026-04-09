# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM3 Causal LM model loader implementation.

SmolLM3-3B: decoder-only causal LM with GQA (4 KV heads) + mixed NoPE/RoPE attention.
Every 4th layer (idx % 4 == 3) uses no positional embedding (NoPE).
Tied input/output embeddings.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
from ....tools.utils import get_static_cache_decode_inputs


class ModelVariant(StrEnum):
    """Available SmolLM3 model variants for causal language modeling."""

    SMOLLM3_3B_INSTRUCT = "3B_Instruct"
    SMOLLM3_3B_BASE = "3B_Base"


class ModelLoader(ForgeModel):
    """SmolLM3 model loader implementation for causal language modeling tasks.

    Architecture highlights:
    - GQA with 16 Q heads and 4 KV heads (ratio 4:1)
    - Mixed NoPE/RoPE: every 4th layer has no positional embedding
    - Tied input/output embeddings (lm_head.weight aliases embed_tokens.weight)
    - SiLU activation in MLP
    - 36 hidden layers, hidden_size=2048, intermediate_size=11008
    """

    _VARIANTS = {
        ModelVariant.SMOLLM3_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM3-3B",
            max_length=128,
        ),
        ModelVariant.SMOLLM3_3B_BASE: LLMModelConfig(
            pretrained_model_name="HuggingFaceTB/SmolLM3-3B-Base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLLM3_3B_INSTRUCT

    sample_text = "Explain how grouped-query attention reduces KV cache memory."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        group = ModelGroup.RED

        return ModelInfo(
            model="SmolLM3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache)."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = getattr(self._variant_config, "max_length", None) or 128
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def load_inputs_prefill(self, dtype_override=None, batch_size=1, seq_len=128):
        """Load prefill-step inputs for SmolLM3."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Generate text long enough for the requested seq_len
        base_text = self.sample_text + " " + "The quick brown fox jumps over the lazy dog. " * 50
        texts = [base_text] * batch_size

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

        self.seq_len = seq_len
        return inputs

    def get_mesh_config(self, num_devices: int):
        # SmolLM3 has 16 Q heads and 4 KV heads
        if num_devices == 32:
            mesh_shape = (8, 4)
        elif self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
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

        # Tied embeddings: lm_head.weight aliases embed_tokens.weight
        # Only shard lm_head; embedding table follows automatically
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
