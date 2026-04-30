# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3 4-layers MTP FP8 model loader implementation.

A compact 4-layer variant of DeepSeek-V3 with FP8 quantization and
Multi-Token Prediction, intended for CI testing.
"""

import sys
import types

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


def _stub_triton_if_missing():
    # transformers finegrained_fp8.py imports triton at module level. On non-CUDA
    # hardware the quantizer sets dequantize=True so actual triton kernels are never
    # called, but the import must succeed. torch._dynamo also accesses triton.language
    # attributes when it is imported. This stub satisfies both without requiring the
    # NVIDIA-only triton package.
    if "triton" not in sys.modules:
        tl = types.ModuleType("triton.language")
        tl.constexpr = type("constexpr", (), {})  # used as annotation; needs to be a type
        tl.dtype = type("dtype", (), {})           # accessed by torch._dynamo.utils
        triton_mod = types.ModuleType("triton")
        triton_mod.jit = lambda fn: fn
        triton_mod.language = tl
        triton_mod.cdiv = lambda a, b: (a + b - 1) // b
        triton_mod.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        triton_mod.__version__ = "0.0.0"
        sys.modules["triton"] = triton_mod
        sys.modules["triton.language"] = tl


def _patch_fp8_dequantize():
    # Fp8Dequantize.convert requires weight dims to be exactly divisible by the
    # block size (128x128). DeepSeek-V3 MLA attention uses kv_a_proj_with_mqa of
    # shape (576, 2560) where 576 % 128 != 0. The stored scale already has the
    # correct ceil-divided block count, so padding the weight before dequantizing
    # and unpadding afterwards is the correct fix.
    import torch

    try:
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    except ImportError:
        return

    def _convert(self, input_dict, full_layer_name=None, **kwargs):
        if len(input_dict) < 2:
            return {full_layer_name: input_dict["weight$"]}

        quantized = input_dict["weight$"][0]
        scales = input_dict["weight_scale_inv"][0]

        rows, cols = quantized.shape[-2:]
        block_size = self.hf_quantizer.quantization_config.weight_block_size
        if block_size is None:
            block_size = (rows, cols)
        block_m, block_n = block_size

        # Derive block counts from the stored scale (supports non-aligned dims).
        n_rows_b, n_cols_b = scales.shape[-2], scales.shape[-1]
        pad_rows = n_rows_b * block_m
        pad_cols = n_cols_b * block_n

        w = quantized.to(scales.dtype)

        if pad_rows != rows or pad_cols != cols:
            buf = torch.zeros(*w.shape[:-2], pad_rows, pad_cols,
                              dtype=w.dtype, device=w.device)
            buf[..., :rows, :cols] = w
            w = buf

        reshaped = w.reshape(-1, n_rows_b, block_m, n_cols_b, block_n)
        sc = scales.reshape(-1, n_rows_b, n_cols_b).unsqueeze(-1).unsqueeze(2)
        dequantized = (reshaped * sc).reshape(*w.shape)

        return {full_layer_name: dequantized[..., :rows, :cols].contiguous()}

    Fp8Dequantize.convert = _convert


def _patch_dynamic_cache():
    # transformers 5.x removed DynamicCache.get_usable_length; the remote
    # modeling_deepseek.py was written against the older API.
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)


def _patch_deepseek_v3_router():
    # DeepseekV3MoE.route_tokens_to_experts uses group_mask.scatter_(1, group_idx, 1)
    # to build a binary group-selection mask. stablehlo.scatter → ttir.scatter
    # produces scatter_reduce_type=invalid for this assign-semantics scatter.
    # Replace with a comparison-based mask (group_idx == arange) which uses
    # only element-wise ops and avoids scatter entirely.
    import torch

    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    except ImportError:
        return

    def _route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]

        # Original: group_mask.scatter_(1, group_idx, 1) — comparison-based equivalent
        # group_idx: [T, topk_group]; result [T, n_group] — 1 where selected group
        group_range = torch.arange(self.n_group, device=router_logits.device).reshape(1, 1, -1)
        group_mask = (group_idx.unsqueeze(-1) == group_range).any(dim=1).to(router_logits.dtype)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    DeepseekV3MoE.route_tokens_to_experts = _route_tokens_to_experts


def _patch_deepseek_v3_native_moe():
    # The native transformers DeepseekV3NaiveMoe uses torch.histc (in the
    # grouped_mm path) or nonzero()+for-loop (in the naive path). Both are
    # data-dependent and break TT XLA compilation.
    # Replace with a fully static all-experts-at-once computation:
    # apply every expert to every token and combine with routing weights.
    # Numerically equivalent (same weighted sum) but no data-dependent control flow.
    import torch

    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3NaiveMoe
    except ImportError:
        return

    def _static_forward(self, hidden_states, top_k_index, top_k_weights):
        n_tokens = hidden_states.shape[0]
        n_experts = self.num_experts

        # gate_up_proj: [E, 2*I, H] — linear weights stored transposed
        # [E, 2I, H] @ [H, T] = [E, 2I, T] → permute → [E, T, 2I]
        gate_up = torch.matmul(
            self.gate_up_proj.to(hidden_states.dtype),
            hidden_states.t(),
        ).permute(0, 2, 1)

        int_dim = gate_up.shape[-1] // 2
        gate = gate_up[..., :int_dim]
        up = gate_up[..., int_dim:]
        activated = self.act_fn(gate) * up  # [E, T, I]

        # down_proj: [E, H, I] — [E, H, I] @ [E, I, T] → [E, H, T] → [E, T, H]
        output = torch.matmul(
            self.down_proj.to(hidden_states.dtype),
            activated.permute(0, 2, 1),
        ).permute(0, 2, 1)  # [E, T, H]

        # Build routing matrix [T, E] via comparison — avoids scatter/histc ops.
        # top_k_index: [T, k], expert_range: [1, 1, E]
        # match: [T, k, E] — 1 where token t's k-th choice is expert e
        expert_range = torch.arange(n_experts, device=hidden_states.device).reshape(1, 1, n_experts)
        match = (top_k_index.unsqueeze(-1) == expert_range).to(hidden_states.dtype)
        routing = (top_k_weights.to(hidden_states.dtype).unsqueeze(-1) * match).sum(dim=1)  # [T, E]

        # Weighted sum: [E, T, H] * [E, T, 1] → sum over E → [T, H]
        return (output * routing.t().unsqueeze(-1)).sum(dim=0).to(hidden_states.dtype)

    DeepseekV3NaiveMoe.forward = _static_forward


class ModelLoader(ForgeModel):
    """DeepSeek V3 4-layers MTP FP8 model loader for causal language modeling."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "ZixiQi/DeepSeek-V3-4layers-MTP-FP8"
        self.tokenizer = None
        self.text = "What is machine learning?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3-4layers-MTP-FP8",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _stub_triton_if_missing()
        _patch_fp8_dequantize()
        _patch_dynamic_cache()
        _patch_deepseek_v3_router()
        _patch_deepseek_v3_native_moe()

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # n_group=8 with n_routed_experts=8 gives 1 expert per group, but the
        # MoE routing uses topk(2) per group which requires at least 2.
        experts_per_group = config.n_routed_experts // config.n_group
        if experts_per_group < 2:
            config.n_group = config.n_routed_experts // 2
            config.topk_group = min(config.topk_group, config.n_group)

        model_kwargs = {
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=config, **model_kwargs
        )

        # Fp8Dequantize outputs float32; cast the whole model so weights and
        # activations share the same dtype.
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
