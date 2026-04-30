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
    # called, but the import must succeed. This stub satisfies the import without
    # requiring the NVIDIA-only triton package.
    if "triton" not in sys.modules:
        tl = types.ModuleType("triton.language")
        tl.constexpr = None  # used as function annotation; any Python object is valid
        triton_mod = types.ModuleType("triton")
        triton_mod.jit = lambda fn: fn
        triton_mod.language = tl
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


def _patch_deepseek_moe(model):
    # DeepseekV3MoE.moe_infer uses tokens_per_expert.cpu().numpy() + a Python
    # for-loop with data-dependent slice bounds, which breaks TT XLA compilation.
    # Replace with a fully static batched computation: apply every expert to every
    # token and combine with the routing weights. Numerically equivalent (same
    # weighted sum) but no data-dependent control flow.
    import torch

    def _batched_moe_infer(self, x, topk_ids, topk_weight):
        n_tokens = x.shape[0]
        n_experts = len(self.experts)
        # [T, E] routing weights; non-zero only at selected (token, expert) pairs
        routing = x.new_zeros(n_tokens, n_experts)
        routing.scatter_add_(1, topk_ids, topk_weight.to(x.dtype))
        # [E, T, d_model] from all experts, then weight and reduce over experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        return (expert_outputs * routing.t().unsqueeze(-1)).sum(dim=0)

    for module in model.modules():
        if type(module).__name__ == "DeepseekV3MoE":
            module.moe_infer = types.MethodType(_batched_moe_infer, module)


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

        _patch_deepseek_moe(model)

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
