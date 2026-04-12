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
    """Available Phi-4 model variants for causal LM."""

    PHI_4 = "Phi_4"


class ModelLoader(ForgeModel):
    """Phi-4 model loader for causal language modeling with tensor parallelism support.

    Phi-4 uses fused projections: qkv_proj (combined Q/K/V) and gate_up_proj
    (combined gate + up). Sharding follows the fused-projection convention.
    """

    _VARIANTS = {
        ModelVariant.PHI_4: LLMModelConfig(
            pretrained_model_name="microsoft/phi-4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHI_4

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
            model="Phi-4",
            variant=variant,
            group=ModelGroup.RED,
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

        Phi-4 uses fused projections:
          - self_attn.qkv_proj: combined Q/K/V — shard output dim on "model"
          - mlp.gate_up_proj: combined gate+up — shard output dim on "model"
          - self_attn.o_proj / mlp.down_proj: shard input dim on "model"

        Args:
            model: Loaded model instance whose weights are to be sharded.
            strategy: "fsdp" shards across both mesh axes; "megatron" shards
                      on the "model" axis only (other axis replicated).
            batch_axis: Non-model mesh axis name for FSDP specs. Pass "data"
                        when input sharding uses a "data"-named mesh axis.

        Returns:
            Dict mapping weight tensors to shard spec tuples.
        """
        shard_specs = {}

        if strategy == "fsdp":
            shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
            shard_specs[model.lm_head.weight] = ("model", batch_axis)
            shard_specs[model.model.norm.weight] = (batch_axis,)
            for layer in model.model.layers:
                # Fused QKV: shard output (first) dim on "model"
                shard_specs[layer.self_attn.qkv_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
                # Fused gate+up: shard output (first) dim on "model"
                shard_specs[layer.mlp.gate_up_proj.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")
                shard_specs[layer.input_layernorm.weight] = (batch_axis,)
                shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

        elif strategy == "megatron":
            shard_specs[model.model.embed_tokens.weight] = (None, None)
            shard_specs[model.lm_head.weight] = ("model", None)
            shard_specs[model.model.norm.weight] = (None,)
            for layer in model.model.layers:
                shard_specs[layer.self_attn.qkv_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
                shard_specs[layer.mlp.gate_up_proj.weight] = ("model", None)
                shard_specs[layer.mlp.down_proj.weight] = (None, "model")
                shard_specs[layer.input_layernorm.weight] = (None,)
                shard_specs[layer.post_attention_layernorm.weight] = (None,)

        else:
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")

        return shard_specs
