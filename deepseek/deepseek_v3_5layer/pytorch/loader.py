# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3 5-layer model loader implementation.

A compact 5-layer variant of DeepSeek-V3 intended for CI testing.
"""

import sys
import transformers

# transformers 5.x removed is_torch_fx_available; the remote modeling_deepseek.py still imports it.
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)

# transformers 5.x removed DynamicCache.get_usable_length; DeepSeek-V3 custom code calls it.
if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = _get_usable_length


def _correct_fp8_weights(model, model_name: str) -> None:
    """Apply block-wise scale correction to FP8 weights that were loaded with unity scale.

    When the FP8 quantization config is stripped, transformers loads float8_e4m3fn checkpoint
    tensors via a plain .to(bfloat16) cast (scale=1), yielding values in [-448, 448] that are
    ~1000x larger than the true weights.  This function reloads the corresponding weight_scale_inv
    tensors (small float32 tensors) and multiplies each 128x128 block by its per-block scale.
    """
    import json
    import os

    import torch
    from safetensors import safe_open
    from transformers.utils import cached_file

    bk, bn = 128, 128

    try:
        index_path = cached_file(
            model_name, "model.safetensors.index.json", local_files_only=True
        )
    except Exception:
        return

    snapshot_dir = os.path.dirname(index_path)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Map: param_name -> scale_inv_name (only for params that have a scale counterpart)
    param_to_scale = {
        k: k.replace(".weight", ".weight_scale_inv")
        for k in weight_map
        if k.endswith(".weight")
        and k.replace(".weight", ".weight_scale_inv") in weight_map
    }

    param_dict = dict(model.named_parameters())

    def _apply_block_scale(param: "torch.Tensor", scale_inv: "torch.Tensor") -> None:
        K, N = param.shape
        K_padded = ((K + bk - 1) // bk) * bk
        N_padded = ((N + bn - 1) // bn) * bn
        nk, nn = K_padded // bk, N_padded // bn

        f = param.float()
        if K_padded != K or N_padded != N:
            padded = torch.zeros(K_padded, N_padded, dtype=f.dtype)
            padded[:K, :N] = f
            f = padded

        f = f.reshape(nk, bk, nn, bn)
        s = scale_inv.to(f.dtype).reshape(nk, 1, nn, 1)
        result = (f * s).reshape(K_padded, N_padded)
        if K_padded != K or N_padded != N:
            result = result[:K, :N]
        param.data.copy_(result.to(torch.bfloat16))

    # Group scale tensors by shard file for efficient loading
    scale_by_shard: dict = {}
    for param_name, scale_name in param_to_scale.items():
        if param_name not in param_dict:
            continue
        shard = weight_map.get(scale_name)
        if shard:
            scale_by_shard.setdefault(shard, []).append((param_name, scale_name))

    for shard_file, pairs in scale_by_shard.items():
        with safe_open(os.path.join(snapshot_dir, shard_file), framework="pt") as st:
            for param_name, scale_name in pairs:
                param = param_dict.get(param_name)
                if param is None or param.ndim != 2:
                    continue
                if scale_name not in st.keys():
                    continue
                scale_inv = st.get_tensor(scale_name)
                _apply_block_scale(param, scale_inv)


def _patch_moe_layers(model) -> None:
    """Replace moe_infer with a static per-expert masked matmul (no numpy / no device transfer)."""
    import torch

    def _static_moe_infer(self_moe, x, topk_ids, topk_weight):
        out = torch.zeros_like(x)
        for i in range(self_moe.experts_per_rank):
            mask = topk_ids == i
            weight = (mask * topk_weight.to(x.dtype)).sum(dim=-1, keepdim=True)
            expert_out = self_moe.experts[i](x)
            out = out + expert_out * weight
        return out

    for module in model.modules():
        if hasattr(module, "moe_infer") and hasattr(module, "experts_per_rank"):
            type(module).moe_infer = _static_moe_infer
            break


class ModelLoader(ForgeModel):
    """DeepSeek V3 5-layer model loader for causal language modeling."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "chwan/DeepSeek-V3-5layer"
        self.tokenizer = None
        self.text = "What is machine learning?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3-5layer",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Strip FP8 quantization config: triton (required by transformers' finegrained_fp8
        # module) is unavailable in this environment.  We load in bfloat16 and apply the
        # block-wise scale correction manually after loading.
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

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
        model.eval()

        # Correct FP8 weights: the plain .to(bfloat16) cast above ignored the per-block
        # scale factors, so all linear weights are ~1000x too large.  Reload the small
        # scale_inv tensors from the checkpoint and multiply each block.
        _correct_fp8_weights(model, self.model_name)

        # Patch MoE dispatch: moe_infer uses .cpu().numpy() on token counts (PJRT
        # device-to-host transfer) and dynamic-length token slices (non-traceable).
        _patch_moe_layers(model)

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
