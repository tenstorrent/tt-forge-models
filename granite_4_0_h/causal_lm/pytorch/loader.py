# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 Hybrid model loader implementation for causal language modeling.
"""

import types
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Uses per-expert masked matmul instead of split-by-expert-size.

    The original GraniteMoeHybridParallelExperts.forward calls
    inputs.split(expert_size) with a Python list, requiring a device-to-host
    transfer that fails on TT silicon.

    For each expert e (statically unrolled at trace time), compute F.linear for
    all tokens and zero-out tokens not assigned to e via a boolean mask.  The
    per-expert weight slices self.weight[e] are constant integer-indexed and
    never create a dynamic gather/embedding.  All ops stay in tensor-land with
    no Python-level splits or device-to-host transfers.
    """
    T = inputs.shape[0]
    result = torch.zeros(T, self.output_size, dtype=inputs.dtype, device=inputs.device)
    for e in range(self.num_experts):
        w_e = self.weight[e]  # [output_size, input_size] — static slice
        out_e = F.linear(inputs, w_e)  # [T, output_size]
        mask_e = (sorted_expert_ids == e).to(inputs.dtype).unsqueeze(1)
        result = result + out_e * mask_e
    return result


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
    """Available Granite 4.0 Hybrid model variants."""

    GRANITE_4_0_H_SMALL = "4.0_H_Small"
    GRANITE_4_0_H_SMALL_BASE = "4.0_H_Small_Base"


class ModelLoader(ForgeModel):
    """Granite 4.0 Hybrid model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_SMALL: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-small",
            max_length=128,
        ),
        ModelVariant.GRANITE_4_0_H_SMALL_BASE: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-small-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_SMALL

    sample_text = "What is the capital of France?"

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
            model="Granite 4.0 Hybrid",
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
        if self.tokenizer.pad_token is None:
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

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

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
