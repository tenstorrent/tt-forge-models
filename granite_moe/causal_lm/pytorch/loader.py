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

    Fix: dense-over-experts forward that never calls .tolist(), index_add, sort,
    or integer comparisons on-device.  Per-expert gate weights are derived via
    floating-point arithmetic (abs-diff < 0.5) to avoid int64/int32 equality
    comparisons, which produce wrong results on TT silicon.
    """
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

    def _patched_moe_forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        x = layer_input.reshape(-1, emb_size)  # [T, H]

        # Router: compute gating without .tolist() (avoids PJRT D2H transfer).
        logits = self.router.layer(x).float()
        top_k_logits, top_k_indices = logits.topk(self.router.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)
        # top_k_indices: [T, K] int64, top_k_gates: [T, K]

        # Build gate_matrix [T, E] without integer comparison on-device.
        # Integers 0..E-1 are exactly representable in bf16 (E ≤ 64 for any
        # GraniteMoE checkpoint, and bf16 has 7 mantissa bits → exact up to 128).
        # diff[j,k,e] = 0.0 when token j's k-th choice is expert e; ≥ 1.0 otherwise.
        expert_ids = torch.arange(
            self.router.num_experts, dtype=x.dtype, device=x.device
        )  # [E]
        top_k_indices_fp = top_k_indices.to(x.dtype).unsqueeze(-1)  # [T, K, 1]
        diff = torch.abs(top_k_indices_fp - expert_ids)  # [T, K, E]
        indicator = (diff < 0.5).to(x.dtype)  # [T, K, E]; 1.0 on match, 0.0 otherwise
        gate_matrix = (top_k_gates.unsqueeze(-1) * indicator).sum(1)  # [T, E]

        # Dense-over-experts: run each expert for ALL tokens and gate the output
        # to 0 for tokens not routed to that expert.  Avoids sort, gather with
        # dynamic indices, and scatter (index_add), all of which misbehave on TT.
        output = torch.zeros_like(x)  # [T, H]
        for e in range(self.router.num_experts):
            gate_e = gate_matrix[:, e]  # [T]
            h_in = F.linear(x, self.input_linear.weight[e])  # [T, 2*I]
            h1, h2 = h_in.chunk(2, dim=-1)
            h_act = self.activation(h1) * h2  # [T, I]
            out = F.linear(h_act, self.output_linear.weight[e])  # [T, H]
            output = output + out * gate_e.unsqueeze(-1)

        return output.view(bsz, length, self.input_size)

    GraniteMoeMoE.forward = _patched_moe_forward
