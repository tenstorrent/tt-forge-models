# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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


class ModelVariant(StrEnum):
    """Available Llama model variants for causal LM."""

    LLAMA_3_1_8B = "3.1_8B"


class ModelLoader(ForgeModel):
    """Llama 3.1 8B model loader for causal language modeling with tensor parallelism support."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_1_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_1_8B

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Llama",
            variant=variant,
            group=ModelGroup.GENERALITY,
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def get_mesh_config(self, num_devices: int):
        if self.config is None:
            self.load_config()
        heads = self.config.num_attention_heads
        if num_devices == 32:
            mesh_shape = (4, 8) if heads % 8 == 0 else (8, 4)
        elif heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif num_devices % 2 == 0 and heads % (num_devices // 2) == 0:
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot split {heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """Return per-tensor shard specs for tensor parallelism.

        Args:
            model: Loaded model instance whose weights are to be sharded.
            strategy: "fsdp" shards across both mesh axes; "megatron" shards
                      on the "model" axis only (other axis replicated).
            batch_axis: Non-model mesh axis name for FSDP specs. Pass "data"
                        when input sharding uses a "data"-named mesh axis.

        Returns:
            Dict mapping weight tensors to shard spec tuples, or None for
            variants that fit comfortably on a single device.
        """
        shard_specs = {}

        if strategy == "fsdp":
            shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
            shard_specs[model.lm_head.weight] = ("model", batch_axis)
            shard_specs[model.model.norm.weight] = (batch_axis,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.gate_proj.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.up_proj.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")
                shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
                shard_specs[layer.input_layernorm.weight] = (batch_axis,)
                shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

        elif strategy == "megatron":
            shard_specs[model.model.embed_tokens.weight] = (None, None)
            shard_specs[model.lm_head.weight] = ("model", None)
            shard_specs[model.model.norm.weight] = (None,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
                shard_specs[layer.mlp.up_proj.weight] = ("model", None)
                shard_specs[layer.mlp.down_proj.weight] = (None, "model")
                shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
                shard_specs[layer.input_layernorm.weight] = (None,)
                shard_specs[layer.post_attention_layernorm.weight] = (None,)

        else:
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")

        return shard_specs
