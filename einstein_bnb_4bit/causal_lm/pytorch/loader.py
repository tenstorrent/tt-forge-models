# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Einstein v6.1 Llama3-8B BnB 4-bit model loader implementation for causal language modeling.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
from ....tools.utils import cast_input_to_type


def _dequantize_bnb4_to_bf16(model: nn.Module, dtype=torch.bfloat16) -> nn.Module:
    """Replace all BNB Linear4bit layers with dequantized nn.Linear.

    bitsandbytes Params4bit.detach() returns a plain Tensor, so model.to(xla_device)
    fails with a RuntimeError. Dequantizing to bfloat16 makes the model moveable to
    any device including TT XLA.

    bitsandbytes is imported lazily here so the loader module can be imported at
    test-collection time before bitsandbytes is installed.
    """
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        dq_weight = bnb_F.dequantize_4bit(
            module.weight.data,
            module.weight.quant_state,
            quant_type=module.weight.quant_type,
        ).to(dtype)
        new_linear = nn.Linear(
            dq_weight.shape[1],
            dq_weight.shape[0],
            bias=module.bias is not None,
            dtype=dtype,
        )
        new_linear.weight = nn.Parameter(dq_weight)
        if module.bias is not None:
            new_linear.bias = nn.Parameter(module.bias.to(dtype))
        setattr(parent, attr, new_linear)
    return model


class ModelVariant(StrEnum):
    """Available Einstein BnB 4-bit model variants for causal LM."""

    EINSTEIN_V6_1_LLAMA3_8B_BNB_4BIT = "v6.1_Llama3_8B_BnB_4bit"


class ModelLoader(ForgeModel):
    """Einstein v6.1 Llama3-8B BnB 4-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EINSTEIN_V6_1_LLAMA3_8B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EINSTEIN_V6_1_LLAMA3_8B_BNB_4BIT

    sample_text = "What is the color of prunes?"

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
            model="Einstein BnB 4-bit",
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
        model_kwargs["device_map"] = "cpu"
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = _dequantize_bnb4_to_bf16(model, dtype=dtype_override or torch.bfloat16)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        input_text = prompt or self.sample_text
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs
