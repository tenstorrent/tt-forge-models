# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GraniteMoeHybrid model loader implementation for causal language modeling.
"""

import types
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


def _patched_topk_gating_forward(self, hidden_states):
    """Avoids expert_size.tolist() by returning sorted_expert_ids as a tensor.

    The original GraniteMoeHybridTopKGating.forward calls expert_size.tolist()
    which triggers a device-to-host transfer that fails on TT silicon (INTERNAL
    error code 13). Instead we return sorted_expert_ids as an int32 tensor so
    the caller (GraniteMoeHybridParallelExperts) can use it with weight indexing.
    """
    logits = self.layer(hidden_states).float()
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
    top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")

    # Return sorted_expert_ids as int32 tensor instead of expert_size list.
    sorted_expert_ids = top_k_experts[index_sorted_experts].int()

    top_k_gates = top_k_gates.flatten()
    batch_gates = top_k_gates[index_sorted_experts]

    return index_sorted_experts, batch_index, batch_gates, sorted_expert_ids, logits


def _patched_parallel_experts_forward(self, inputs, sorted_expert_ids):
    """Uses matmul + one-hot masking instead of split-by-expert-size or gather.

    The original GraniteMoeHybridParallelExperts.forward calls
    inputs.split(expert_size) with a Python list, requiring a device→host
    transfer that fails on TT silicon.

    A simple weight gather (self.weight[sorted_expert_ids]) is also
    problematic: MLIR flattens the 3D weight [num_experts, output_size,
    input_size] to a 2D embedding table [num_experts, output_size*input_size],
    whose row size (~3 MB) overflows the L1 CB budget on TT silicon.

    Instead: flatten weight to [num_experts*output_size, input_size], matmul
    against all inputs at once, reshape the result to
    [T, num_experts, output_size], then mask with a per-token one-hot to
    select each token's assigned expert.  All operations stay in tensor-land
    with no Python-level splits or device→host transfers.
    """
    T = inputs.shape[0]

    # [num_experts * output_size, input_size]
    w = self.weight.view(-1, self.input_size)

    # [T, num_experts * output_size]
    all_outputs = inputs @ w.T

    # [T, num_experts, output_size]
    all_outputs = all_outputs.view(T, self.num_experts, self.output_size)

    # One-hot selector [T, num_experts] — comparison avoids int64 / tolist.
    expert_range = torch.arange(
        self.num_experts, dtype=sorted_expert_ids.dtype, device=inputs.device
    )
    one_hot = (sorted_expert_ids.unsqueeze(1) == expert_range.unsqueeze(0)).to(
        inputs.dtype
    )

    # [T, output_size]
    return (all_outputs * one_hot.unsqueeze(2)).sum(dim=1)


def _patch_moe_experts(model):
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridParallelExperts,
        GraniteMoeHybridTopKGating,
    )

    for module in model.modules():
        if isinstance(module, GraniteMoeHybridTopKGating):
            module.forward = types.MethodType(_patched_topk_gating_forward, module)
        elif isinstance(module, GraniteMoeHybridParallelExperts):
            module.forward = types.MethodType(
                _patched_parallel_experts_forward, module
            )


class ModelVariant(StrEnum):
    """Available GraniteMoeHybrid model variants."""

    TINY_RANDOM = "tiny_random"
    GRANITE_4_0_H_TINY = "granite_4_0_h_tiny"


class ModelLoader(ForgeModel):
    """GraniteMoeHybrid model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-granitemoehybrid",
        ),
        ModelVariant.GRANITE_4_0_H_TINY: LLMModelConfig(
            pretrained_model_name="unsloth/granite-4.0-h-tiny",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GraniteMoeHybrid",
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

        _patch_moe_experts(model)

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

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        max_length = getattr(self._variant_config, "max_length", None)
        if max_length is not None:
            padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], max_length)
            padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], max_length)
            self.seq_len = seq_len
            inputs["input_ids"] = padded_input_ids
            inputs["attention_mask"] = padded_attention_mask

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
