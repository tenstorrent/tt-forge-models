# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Crystal model loader implementation for causal language modeling.
"""

import sys
import types
from typing import Optional, Set, Tuple

import torch
import transformers.pytorch_utils as _pt_utils
import transformers.utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


def _find_pruneable_heads_and_indices(
    heads: list,
    n_heads: int,
    head_size: int,
    already_pruned_heads: set,
) -> Tuple[Set[int], torch.LongTensor]:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def _prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    b = layer.bias.clone().detach() if dim == 0 else layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def _get_device_map(n_layers, devices):
    layers_per_device = n_layers // len(devices)
    remainder = n_layers % len(devices)
    device_map = {}
    layer_idx = 0
    for i, device in enumerate(devices):
        n = layers_per_device + (1 if i < remainder else 0)
        for _ in range(n):
            device_map[layer_idx] = device
            layer_idx += 1
    return device_map


def _assert_device_map(device_map, n_layers):
    if set(device_map.keys()) != set(range(n_layers)):
        raise ValueError(
            f"device_map must cover all {n_layers} layers, got keys {set(device_map.keys())}"
        )


# Restore symbols removed in transformers 5.x that Crystal's custom code needs.
if not hasattr(_pt_utils, "find_pruneable_heads_and_indices"):
    _pt_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
if not hasattr(_pt_utils, "prune_conv1d_layer"):
    _pt_utils.prune_conv1d_layer = _prune_conv1d_layer

_model_parallel_mod_name = "transformers.utils.model_parallel_utils"
if _model_parallel_mod_name not in sys.modules:
    _compat_mod = types.ModuleType(_model_parallel_mod_name)
    _compat_mod.get_device_map = _get_device_map
    _compat_mod.assert_device_map = _assert_device_map
    sys.modules[_model_parallel_mod_name] = _compat_mod
    transformers.utils.model_parallel_utils = _compat_mod

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Crystal model variants."""

    CRYSTAL = "Crystal"


class ModelLoader(ForgeModel):
    """Crystal model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.CRYSTAL: LLMModelConfig(
            pretrained_model_name="LLM360/Crystal",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CRYSTAL

    sample_text = "int add(int x, int y) {"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Crystal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            config.n_layer = self.num_layers
        # add_cross_attention was removed from PretrainedConfig in transformers 5.x
        if not hasattr(config, "add_cross_attention"):
            config.add_cross_attention = False
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
