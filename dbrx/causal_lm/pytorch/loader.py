# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DBRX model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available DBRX model variants for causal language modeling."""

    TINY_RANDOM = "tiny-random"


def _patched_dbrx_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Replace nonzero/for-loop MoE dispatch with a static loop over all experts.

    The original DbrxExperts.forward calls nonzero() to find active experts,
    then iterates over a dynamically-sized tensor. Both operations force a
    device-to-host transfer on TT silicon (PJRT INTERNAL error 13).

    This replacement loops over all num_experts statically and uses boolean
    masking to zero out contributions from non-selected experts, keeping the
    entire computation graph static and traceable by torch.compile / XLA.
    """
    batch_size = hidden_states.shape[0]
    # [S, d_model] where S = batch * seq_len
    flat_hidden = hidden_states.reshape(-1, self.ffn_hidden_size)
    S = flat_hidden.shape[0]

    next_states = torch.zeros(
        S, self.ffn_hidden_size, dtype=flat_hidden.dtype, device=flat_hidden.device
    )

    split_expert_shape = (-1, self.ffn_hidden_size, self.hidden_size)

    for expert_idx in range(self.num_experts):
        # Boolean mask: [S, top_k] — True where this expert is chosen
        expert_bool = top_k_index == expert_idx  # [S, top_k]

        # Per-token weight for this expert; 0 for tokens not routed here
        token_weights = (top_k_weights * expert_bool.to(top_k_weights.dtype)).sum(
            dim=-1
        )  # [S]

        v1 = self.mlp.v1.view(split_expert_shape)[expert_idx]
        w1 = self.mlp.w1.view(split_expert_shape)[expert_idx]
        w2 = self.mlp.w2.view(split_expert_shape)[expert_idx]

        states = self.mlp(flat_hidden, w1, v1, w2)  # [S, ffn_hidden_size]
        states = states * token_weights[:, None]
        next_states = next_states + states

    next_states = next_states.view(batch_size, -1, self.ffn_hidden_size)
    return next_states


def _patch_moe_experts(model):
    import types
    from transformers.models.dbrx.modeling_dbrx import DbrxExperts

    for module in model.modules():
        if isinstance(module, DbrxExperts):
            module.forward = types.MethodType(_patched_dbrx_experts_forward, module)


class ModelLoader(ForgeModel):
    """DBRX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="trl-internal-testing/tiny-DbrxForCausalLM",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    sample_text = "This is a sample text from "

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DBRX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        )
        model.eval()

        _patch_moe_experts(model)

        return model

    def load_inputs(self, dtype_override=None):
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

        return inputs
