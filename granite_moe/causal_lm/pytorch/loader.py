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
    """Replace GraniteMoeMoE.forward to eliminate device-to-host transfers and
    stablehlo.scatter with mismatched begins rank.

    Two bugs in the original GraniteMoeMoE forward:
    1. GraniteMoeTopKGating.forward calls expert_size.tolist() → PJRT D2H
       transfer → INTERNAL: Error code: 13.
    2. zeros.index_add(0, batch_index, expert_outputs) lowers to
       stablehlo.scatter; the tt-mlir scatter lowering creates a 2-element
       slice of a 2D index tensor, but TTNN promotes 2D tensors to 4D at
       runtime → TT_FATAL: Input rank 4 and begins 2 must have the same size.

    Fix: replace the full GraniteMoeMoE.forward with a static per-expert
    boolean-mask matmul that never calls .tolist() or index_add.
    """
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

    def _patched_moe_forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        x = layer_input.reshape(-1, emb_size)
        num_tokens = x.size(0)

        # Router: compute routing without .tolist() (avoids PJRT D2H transfer).
        logits = self.router.layer(x).float()
        top_k_logits, top_k_indices = logits.topk(self.router.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)

        # Sort tokens by expert assignment.
        top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)
        batch_index = index_sorted_experts.div(self.router.top_k, rounding_mode="trunc")
        batch_gates = top_k_gates.flatten()[index_sorted_experts]
        sorted_experts = top_k_experts[index_sorted_experts]  # [num_tokens * top_k]

        expert_inputs = x[batch_index]  # [num_tokens * top_k, emb_size]

        # Static per-expert forward: boolean mask per expert so no dynamic split
        # or expert_size.tolist() is needed.  For non-expert positions the input
        # is zeroed, so F.linear output (no bias) is also zero.
        expert_results = []
        for e in range(self.router.num_experts):
            mask = (sorted_experts == e).to(x.dtype).unsqueeze(-1)  # [N*K, 1]
            h_in = F.linear(expert_inputs * mask, self.input_linear.weight[e])
            h1, h2 = h_in.chunk(2, dim=-1)
            # h1/h2 zero for non-expert positions; activation(0)=0 for GELU/SiLU
            h_act = self.activation(h1) * h2
            out = F.linear(h_act, self.output_linear.weight[e])
            expert_results.append(out)

        expert_outputs = sum(expert_results) * batch_gates.unsqueeze(-1)

        # Per-token aggregation: replace index_add (scatter) with masked sum.
        # index_add lowers to stablehlo.scatter which generates a 2-element-begins
        # slice for a tensor that TTNN promotes to 4D, causing a rank mismatch.
        output_rows = []
        for j in range(num_tokens):
            mask_j = (batch_index == j).to(expert_outputs.dtype)  # [N*K]
            row = (expert_outputs * mask_j.unsqueeze(-1)).sum(0, keepdim=True)
            output_rows.append(row)

        layer_output = torch.cat(output_rows, dim=0)
        return layer_output.view(bsz, length, self.input_size)

    GraniteMoeMoE.forward = _patched_moe_forward
