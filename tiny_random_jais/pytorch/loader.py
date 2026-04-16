# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random JAIS model loader implementation for causal language modeling.
"""
import sys
import types

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional


def _patch_transformers_compat():
    """Patch missing transformers modules needed by JAIS custom code.

    The JAIS model uses custom code written for older transformers versions.
    This patches:
    - find_pruneable_heads_and_indices (removed in transformers 5.x)
    - prune_conv1d_layer (removed in transformers 5.x)
    - transformers.utils.model_parallel_utils (removed in transformers 5.x)
    - PreTrainedModel.get_head_mask (removed in transformers 5.x)
    """
    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                if head not in already_pruned:
                    mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )

    if not hasattr(pytorch_utils, "prune_conv1d_layer"):

        def prune_conv1d_layer(layer, index, dim=0):
            nf = layer.nf
            weight = layer.weight.index_select(dim, index)
            if dim == 0:
                bias = layer.bias.index_select(0, index)
                nf = len(index)
            else:
                bias = layer.bias
            new_layer = pytorch_utils.Conv1D(nf, layer.weight.shape[1 - dim])
            new_layer.weight = torch.nn.Parameter(weight)
            new_layer.bias = torch.nn.Parameter(bias)
            return new_layer

        pytorch_utils.prune_conv1d_layer = prune_conv1d_layer

    if "transformers.utils.model_parallel_utils" not in sys.modules:
        mpu = types.ModuleType("transformers.utils.model_parallel_utils")

        def assert_device_map(device_map, num_blocks):
            pass

        def get_device_map(n_layers, devices):
            return {i: devices[0] for i in range(n_layers)}

        mpu.assert_device_map = assert_device_map
        mpu.get_device_map = get_device_map
        sys.modules["transformers.utils.model_parallel_utils"] = mpu

    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(
            self, head_mask, num_hidden_layers, is_attention_chunked=False
        ):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask


_patch_transformers_compat()

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Tiny Random JAIS model variants."""

    TINY_RANDOM_JAIS = "tiny_random_jais"


class ModelLoader(ForgeModel):
    """Tiny Random JAIS model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_JAIS: ModelConfig(
            pretrained_model_name="katuni4ka/tiny-random-jais",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_JAIS

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TinyRandomJais",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {
            "padding_side": "left",
        }
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "add_cross_attention"):
            config.add_cross_attention = False
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        test_input = "This is a sample text from "

        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            test_input,
            return_tensors="pt",
            max_length=32,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
