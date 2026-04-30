# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EuroLLM model loader implementation for causal language modeling.
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
    """Replace TorchAtenAwqLinear modules with nn.Linear using dequantized fp weights.

    gptqmodel's TorchAtenAwqLinear._fused_op_forward raises NotImplementedError
    for non-CUDA tensors, blocking TT XLA compilation. Dequantize before any
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


class ModelVariant(StrEnum):
    """Available EuroLLM model variants for causal language modeling."""

    EUROLLM_22B_INSTRUCT_2512 = "EuroLLM_22B_Instruct_2512"
    STELTERLAB_EUROLLM_9B_INSTRUCT_AWQ = "stelterlab_EuroLLM_9B_Instruct_AWQ"


class ModelLoader(ForgeModel):
    """EuroLLM model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EUROLLM_22B_INSTRUCT_2512: LLMModelConfig(
            pretrained_model_name="utter-project/EuroLLM-22B-Instruct-2512",
            max_length=256,
        ),
        ModelVariant.STELTERLAB_EUROLLM_9B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="stelterlab/EuroLLM-9B-Instruct-AWQ",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EUROLLM_22B_INSTRUCT_2512

    sample_text = "What is the capital of Portugal? How would you describe it?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EuroLLM",
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
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant == ModelVariant.STELTERLAB_EUROLLM_9B_INSTRUCT_AWQ:
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = _dequantize_awq_layers(model, dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
