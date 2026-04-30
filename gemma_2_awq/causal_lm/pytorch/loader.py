# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 2 AWQ model loader implementation for causal language modeling.
"""
import torch
import torch.nn as nn
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


def _dequantize_awq_layers(model, dtype):
    """Replace AWQ quantized linear layers with standard float nn.Linear layers.

    gptqmodel's TorchAtenAwqLinear uses a CPU-only fused kernel
    (torch.ops.aten._weight_int4pack_mm_for_cpu) and sets stateful
    linear_mode='inference' on the first CPU forward pass.  When the TT
    device runner subsequently calls the compiled model with TT tensors,
    _fused_op_forward raises NotImplementedError because x.device != 'cpu'.
    Dequantizing to float before any forward pass avoids this entirely.
    """
    replacements = {}
    for name, module in model.named_modules():
        if hasattr(module, "awq_weight_dequantize") and hasattr(module, "in_features"):
            # Returns [in_features, out_features]; nn.Linear.weight is [out_features, in_features]
            weight = module.awq_weight_dequantize(device="cpu", dtype=dtype)
            has_bias = getattr(module, "bias", None) is not None
            linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
            linear.weight = nn.Parameter(weight.t().contiguous())
            if has_bias:
                linear.bias = nn.Parameter(module.bias.to(dtype))
            replacements[name] = linear

    for name, linear in replacements.items():
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
        else:
            parent, child_name = model, name
        setattr(parent, child_name, linear)

    return model


class ModelVariant(StrEnum):
    """Available Gemma 2 AWQ model variants for causal language modeling."""

    GEMMA_2_9B_IT_AWQ_INT4 = "9B_IT_AWQ_INT4"


class ModelLoader(ForgeModel):
    """Gemma 2 AWQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_2_9B_IT_AWQ_INT4: LLMModelConfig(
            pretrained_model_name="hugging-quants/gemma-2-9b-it-AWQ-INT4",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_2_9B_IT_AWQ_INT4

    sample_text = "What is your favorite city?"

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
            model="Gemma 2 AWQ",
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
        model_kwargs["device_map"] = "cpu"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = _dequantize_awq_layers(model, target_dtype)

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

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
