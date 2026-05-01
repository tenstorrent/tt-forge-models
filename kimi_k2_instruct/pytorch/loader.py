# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 Instruct FP4 model loader implementation.
"""

import os
import sys
from typing import Optional
from unittest.mock import patch

import torch

# Patch missing functions before importing model code that depends on them.
# The model's remote code was written for an older transformers that included
# these helpers; newer versions removed them.
import transformers.utils
import transformers.utils.import_utils

if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):

    def _is_flash_attn_gte_2_10():
        return False

    transformers.utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_gte_2_10
    sys.modules["transformers.utils"].__dict__[
        "is_flash_attn_greater_or_equal_2_10"
    ] = _is_flash_attn_gte_2_10

if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

# Patch DynamicCache.from_legacy_cache removed in newer transformers
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "from_legacy_cache"):

    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(key, value, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache

if not hasattr(DynamicCache, "to_legacy_cache"):

    def _to_legacy_cache(self):
        legacy_cache = []
        for layer in self.layers:
            legacy_cache.append((layer.keys, layer.values))
        return legacy_cache

    DynamicCache.to_legacy_cache = _to_legacy_cache

if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = _get_usable_length

import transformers.models.gpt2.tokenization_gpt2 as _gpt2_tok

if not hasattr(_gpt2_tok, "bytes_to_unicode"):

    def _bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    _gpt2_tok.bytes_to_unicode = _bytes_to_unicode

import numpy as np
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_imports

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _fixed_moe_infer(self, x, topk_ids, topk_weight):
    # Fix: use .tolist() so loop variables are Python ints, not numpy scalars.
    # Numpy scalars cause AttributeError under Dynamo's FakeTensor tracing when
    # the loop body does integer arithmetic that passes through __torch_function__.
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    sorted_tokens_shape = sorted_tokens.shape
    if self.ep_size > 1:
        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(self.ep_size, -1).sum(1).cpu().tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            self.ep_size, self.experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().tolist()):
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
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelVariant(StrEnum):
    """Available Kimi K2 Instruct FP4 model variants."""

    BASETEN_FP4 = "baseten-FP4"


class ModelLoader(ForgeModel):
    """Kimi K2 Instruct FP4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASETEN_FP4: None,
    }

    DEFAULT_VARIANT = ModelVariant.BASETEN_FP4

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        num_layers: Optional[int] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use.
                        If None, uses a reduced default.
        """
        super().__init__(variant)
        self.model_name = "baseten/Kimi-K2-Instruct-FP4"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Kimi-K2-Instruct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi K2 Instruct FP4 model with reduced config.

        The full model is a 581B-parameter MoE causal LM based on DeepSeek V3
        architecture. We load with a reduced configuration for testing.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.

        Returns:
            torch.nn.Module: The DeepSeek V3 causal LM instance.
        """
        model = None
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = 2
            config.num_attention_heads = 16
            config.hidden_size = 1024
            config.num_key_value_heads = 16
            config.intermediate_size = 1024 * 4
            config.num_experts_per_tok = 2
            config.q_lora_rank = 256
            config.use_flash_attention = False
            config._attn_implementation = "eager"

            model_kwargs = {
                "attn_implementation": "eager",
                "trust_remote_code": True,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model_class = get_class_from_dynamic_module(
                "modeling_deepseek.DeepseekV3ForCausalLM",
                self.model_name,
                trust_remote_code=True,
            )

            # Patch moe_infer: remote code uses .cpu().numpy() which yields numpy
            # scalars that crash Dynamo's FakeTensor evaluation in the loop body.
            for _mod_name, _mod in list(sys.modules.items()):
                if "modeling_deepseek" in _mod_name and hasattr(_mod, "DeepseekV3MoE"):
                    _mod.DeepseekV3MoE.moe_infer = _fixed_moe_infer
                    break

            model = model_class(config)
            model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Kimi K2 Instruct FP4 model.

        Args:
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
