# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.1 Terminus model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 671B parameter
model is too large to load directly.
"""

import os
import sys
from typing import Optional
from unittest.mock import patch

import torch
import transformers
import transformers.utils.import_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_imports

# transformers 5.x removed is_torch_fx_available from import_utils; remote
# model code (modeling_deepseek.py) imports it at module level.
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

# transformers 5.x removed DynamicCache.from_legacy_cache; remote model code
# calls it when past_key_values is not already a Cache instance.
if not hasattr(DynamicCache, "from_legacy_cache"):

    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(key, value, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache

# transformers 5.x removed DynamicCache.get_usable_length; for DynamicCache
# it was equivalent to get_seq_length (no static max length).
if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = _get_usable_length

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


def _fixed_moe_infer(self, x, topk_ids, topk_weight):
    """Patched moe_infer: uses .tolist() instead of .numpy() for tokens_per_expert
    so the dispatch loop iterates over Python ints rather than numpy integers.
    Dynamo traces numpy integer arithmetic as tensor ops and fails to evaluate
    them with FakeTensors; Python int arithmetic is handled as Python scalars."""
    import numpy as np
    import torch.distributed as dist

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    sorted_tokens_shape = sorted_tokens.shape
    if self.ep_size > 1:
        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(
            tokens_per_expert.shape[0]
        )
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(self.ep_size, -1)
            .sum(1)
            .cpu()
            .numpy()
            .tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            self.ep_size, self.experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather
    tokens_per_expert = tokens_per_expert.cpu().tolist()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = self.experts[i + self.ep_rank * self.experts_per_rank]
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    if self.ep_size > 1:
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelLoader(ForgeModel):
    """DeepSeek V3.1 Terminus model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-V3.1-Terminus"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V3.1-Terminus",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

            # Reduce model dimensions for testing
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = 6
            config.num_attention_heads = 16
            config.hidden_size = 1024
            config.num_key_value_heads = 16
            config.intermediate_size = 1024 * 4
            config.num_experts_per_tok = 2
            config.q_lora_rank = 256
            config.use_flash_attention = False

            model_kwargs = {
                "attn_implementation": "eager",
                "trust_remote_code": True,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        # Patch DeepseekV3MoE.moe_infer to use Python ints instead of numpy integers
        for mod_name, mod in list(sys.modules.items()):
            if "modeling_deepseek" in mod_name and hasattr(mod, "DeepseekV3MoE"):
                mod.DeepseekV3MoE.moe_infer = _fixed_moe_infer
                break

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
