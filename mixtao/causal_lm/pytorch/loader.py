# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MixTAO model loader implementation for causal language modeling.
"""

import types
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MixTAO model variants."""

    MIXTAO_7BX2_MOE_V8_1 = "7Bx2_MoE_v8.1"


def _patch_mixtral_experts_static(model):
    # Replace grouped_mm expert dispatch (uses torch.histc on int, crashes TT XLA)
    # with a static per-expert masked matmul over range(num_experts).
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    def static_forward(self, hidden_states, top_k_index, top_k_weights):
        weight_dtype = self.gate_up_proj.dtype
        hidden_states = hidden_states.to(weight_dtype)
        top_k_weights = top_k_weights.to(weight_dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            gate_up_out = F.linear(hidden_states, self.gate_up_proj[expert_idx])
            gate, up = gate_up_out.chunk(2, dim=-1)
            expert_out = F.linear(
                self.act_fn(gate).to(weight_dtype) * up, self.down_proj[expert_idx]
            )
            expert_weight = (
                top_k_weights
                * (top_k_index == expert_idx).to(weight_dtype)
            ).sum(dim=-1, keepdim=True)
            final_hidden_states = final_hidden_states + expert_out * expert_weight
        return final_hidden_states

    for module in model.modules():
        if isinstance(module, MixtralExperts):
            module.forward = types.MethodType(static_forward, module)


class ModelLoader(ForgeModel):
    """MixTAO model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MIXTAO_7BX2_MOE_V8_1: ModelConfig(
            pretrained_model_name="mixtao/MixTAO-7Bx2-MoE-v8.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIXTAO_7BX2_MOE_V8_1

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MixTAO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

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
        ).eval()

        _patch_mixtral_experts_static(model)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
