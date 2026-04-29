# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 3.5 AWQ model loader implementation for causal language modeling.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.utils.generic as _tug
from typing import Optional

# transformers.utils.generic.check_model_inputs was added after 5.2.0;
# it is an alias for merge_with_config_defaults (deprecated in 5.7.0).
if not hasattr(_tug, "check_model_inputs"):
    from transformers.utils.generic import merge_with_config_defaults
    _tug.check_model_inputs = merge_with_config_defaults


def _dequantize_awq_layers(model, dtype):
    """Replace TorchAtenAwqLinear modules with nn.Linear using dequantized fp weights.

    gptqmodel's TorchAtenAwqLinear._fused_op_forward raises NotImplementedError
    for non-CPU tensors, blocking TT XLA compilation. Dequantize before any
    forward pass so the model is device-agnostic.
    """
    try:
        from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
    except ImportError:
        return model

    target_dtype = dtype if dtype is not None else torch.bfloat16

    for name, module in list(model.named_modules()):
        if not isinstance(module, TorchAtenAwqLinear):
            continue
        # awq_weight_dequantize returns [in_features, out_features]; nn.Linear
        # stores weight as [out_features, in_features], so transpose.
        W = module.awq_weight_dequantize(device="cpu", dtype=target_dtype)
        has_bias = module.bias is not None
        linear = nn.Linear(module.in_features, module.out_features,
                           bias=has_bias, device="cpu", dtype=target_dtype)
        linear.weight.data = W.T.contiguous()
        if has_bias:
            linear.bias.data = module.bias.data.to(dtype=target_dtype)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], linear)

    return model

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
    """Available EXAONE 3.5 AWQ model variants for causal language modeling."""

    EXAONE_3_5_7_8B_INSTRUCT_AWQ = "3.5_7.8B_Instruct_AWQ"


class ModelLoader(ForgeModel):
    """EXAONE 3.5 AWQ model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_AWQ

    sample_text = "Explain the basics of large language models."

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
            model="EXAONE 3.5 AWQ",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True, "device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        model = _dequantize_awq_layers(model, dtype_override)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "system",
                "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
            },
            {
                "role": "user",
                "content": self.sample_text,
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

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
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
