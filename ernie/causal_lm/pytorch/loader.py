# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ERNIE 4.5 MoE causal language modeling loader
"""
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


def _tt_static_ernie_moe_forward(experts_module, hidden_states, top_k_index, top_k_weights):
    # Avoid torch.histc on Int (grouped_mm, XLA unsupported) and avoid large
    # 3D embedding gather (batched_mm, L1 CB overflow for ERNIE's expert dims).
    # Loop over Python ints so dynamo unrolls into static F.linear calls.
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(experts_module.num_experts):
        mask = (top_k_index == expert_idx)
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(hidden_states, experts_module.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(experts_module.act_fn(gate) * up, experts_module.down_proj[expert_idx])
        out = out + hidden_expert * weight
    return out.to(dtype)


# Register before from_pretrained so get_interface can look it up at forward time.
# __setitem__ adds to the local mapping of the singleton ALL_EXPERTS_FUNCTIONS
# instance which is what the @use_experts_implementation dispatch reads.
ALL_EXPERTS_FUNCTIONS["tt_static_ernie_moe"] = _tt_static_ernie_moe_forward


class ModelVariant(StrEnum):
    """Available ERNIE 4.5 MoE model variants."""

    ERNIE_4_5_21B_A3B_PT = "21B_A3B_PT"


class ModelLoader(ForgeModel):
    """ERNIE 4.5 MoE model loader implementation."""

    _VARIANTS = {
        ModelVariant.ERNIE_4_5_21B_A3B_PT: LLMModelConfig(
            pretrained_model_name="baidu/ERNIE-4.5-21B-A3B-PT",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ERNIE_4_5_21B_A3B_PT

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
            model="ERNIE-4.5-MoE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        # grouped_mm_experts_forward uses torch.histc on Int, unsupported by XLA/TT.
        # batched_mm_experts_forward gathers gate_up_proj[expert_ids] producing a
        # 3D tensor whose row size (3072 × 2560 × 2 B = 15.7 MB) overflows L1.
        # Switch to static per-expert masked matmul registered above.
        # The setter does not validate, so this bypasses the from_pretrained check
        # that only accepts "eager", "grouped_mm", and "batched_mm".
        model.config._experts_implementation = "tt_static_ernie_moe"

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        self._ensure_tokenizer()

        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        max_length = self._variant_config.max_length

        tokenized_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )

        inputs = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
        }

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        self._ensure_tokenizer()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return decoded[0] if len(decoded) == 1 else decoded
