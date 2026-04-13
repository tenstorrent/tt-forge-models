# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mixtral MoE model loader implementation for causal language modeling
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


class ModelVariant(StrEnum):
    """Available Mixtral model variants."""

    MIXTRAL_8X7B_INSTRUCT_V01 = "8x7B_Instruct_v0.1"


class ModelLoader(ForgeModel):
    """Mixtral MoE model loader implementation."""

    _VARIANTS = {
        ModelVariant.MIXTRAL_8X7B_INSTRUCT_V01: LLMModelConfig(
            pretrained_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIXTRAL_8X7B_INSTRUCT_V01

    def __init__(self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mixtral",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        messages = [{"role": "user", "content": "What is the capital of France?"}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        model_name = self._variant_config.pretrained_model_name
        self.config = AutoConfig.from_pretrained(model_name)
        return self.config

    def get_mesh_config(self, num_devices: int):
        if self.config is None:
            self.load_config()

        # 32 attention heads, 8 KV heads — divisible by 2, 4, 8
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
            # Attention: standard Megatron column/row parallel
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            # MoE experts: shard gate_up_proj and down_proj along expert dim
            # MixtralExperts uses pre-stacked fused weights:
            #   gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
            #   down_proj: [num_experts, hidden_size, intermediate_size]
            shard_specs[layer.block_sparse_moe.experts.gate_up_proj] = ("model", None, "batch")
            shard_specs[layer.block_sparse_moe.experts.down_proj] = ("model", "batch", None)

        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs
