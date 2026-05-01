# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite MoE model loader implementation for causal language modeling.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    """Available Granite MoE model variants for causal language modeling."""

    GRANITE_3_0_1B_A400M_BASE = "3.0_1B_A400M_Base"
    GRANITE_3_1_1B_A400M_BASE = "3.1_1B_A400M_Base"


class ModelLoader(ForgeModel):
    """Granite MoE model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_3_0_1B_A400M_BASE: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-3.0-1b-a400m-base",
            max_length=128,
        ),
        ModelVariant.GRANITE_3_1_1B_A400M_BASE: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-3.1-1b-a400m-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_3_0_1B_A400M_BASE

    sample_text = "Where is the Thomas J. Watson Research Center located?"

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
            model="Granite MoE",
            variant=variant,
            group=ModelGroup.VULCAN,
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
        if self.tokenizer.pad_token is None:
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

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        _patch_moe_experts(model)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config


def _patch_moe_experts(model):
    """Replace GraniteMoe MoE dispatch to avoid expert_size.tolist() D2H transfer.

    GraniteMoeTopKGating.forward calls expert_size.tolist() which triggers a
    PJRT device-to-host transfer on TT hardware (INTERNAL: Error code: 13).
    Replace TopKGating and ParallelExperts with static per-expert masked matmul.
    """
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeTopKGating,
        GraniteMoeParallelExperts,
    )

    def _patched_topk_gating_forward(self, hidden_states):
        logits = self.layer(hidden_states).float()
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)

        zeros = torch.zeros(
            [top_k_gates.size(0), self.num_experts],
            dtype=top_k_gates.dtype,
            device=top_k_gates.device,
        )
        gates = zeros.scatter(1, top_k_indices, 1)
        expert_size_tensor = gates.long().sum(0)  # [num_experts] — kept on device

        top_k_experts = top_k_indices.flatten()
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")

        top_k_gates = top_k_gates.flatten()
        batch_gates = top_k_gates[index_sorted_experts]

        # Return sorted expert IDs (tensor) instead of expert_size list so there
        # is no device-to-host transfer.
        sorted_experts = top_k_experts[index_sorted_experts]
        return index_sorted_experts, batch_index, batch_gates, sorted_experts, logits

    def _patched_parallel_experts_forward(self, inputs, sorted_experts):
        # sorted_experts: [num_tokens * top_k] tensor of expert IDs in sorted order.
        # Use a static per-expert boolean-mask matmul; avoids dynamic split (which
        # needs expert_size as a Python list) and any device-to-host transfers.
        output_list = []
        for i in range(self.num_experts):
            mask = (sorted_experts == i).to(inputs.dtype).unsqueeze(-1)
            # Zero out non-expert-i rows before linear; F.linear has no bias so
            # output for zeroed rows is zero.
            expert_out = F.linear(inputs * mask, self.weight[i])
            output_list.append(expert_out)
        return sum(output_list)

    GraniteMoeTopKGating.forward = _patched_topk_gating_forward
    GraniteMoeParallelExperts.forward = _patched_parallel_experts_forward
