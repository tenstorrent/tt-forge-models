# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    QWEN3_0_6B = "0_6B"


class ModelLoader(ForgeModel):
    """Qwen3-0.6B causal LM loader with tensor parallelism support."""

    _VARIANTS = {
        ModelVariant.QWEN3_0_6B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-0.6B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_0_6B

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

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
                f"Cannot split {heads} attention heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """Return per-tensor shard specs for tensor parallelism.

        Only projection matrices and lm_head are sharded; layernorms and
        embedding remain replicated (not in shard_spec).

        Args:
            model: Loaded model instance whose weights are to be sharded.
            strategy: "fsdp" shards across both mesh axes; "megatron" shards
                      on the "model" axis only (other axis replicated).
            batch_axis: Non-model mesh axis label; pass "data" when DP input
                        sharding uses a "data"-named mesh axis.

        Returns:
            Dict mapping weight tensors to shard spec tuples.
        """
        shard_specs = {}

        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.up_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")

        shard_specs[model.lm_head.weight] = ("model", batch_axis)

        return shard_specs
