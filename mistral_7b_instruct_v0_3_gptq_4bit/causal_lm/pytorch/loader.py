# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral 7B Instruct v0.3 GPTQ 4-bit model loader implementation for causal language modeling (PyTorch).
"""

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
)


class ModelVariant(StrEnum):
    """Available Mistral 7B Instruct v0.3 GPTQ 4-bit model variants for causal LM."""

    MISTRAL_7B_INSTRUCT_V0_3_GPTQ_4BIT = "Mistral-7B-Instruct-v0.3-GPTQ-4bit"


class ModelLoader(ForgeModel):
    """Mistral 7B Instruct v0.3 GPTQ 4-bit model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MISTRAL_7B_INSTRUCT_V0_3_GPTQ_4BIT: LLMModelConfig(
            pretrained_model_name="RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_7B_INSTRUCT_V0_3_GPTQ_4BIT

    sample_text = "What is the meaning of life?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Mistral 7B Instruct v0.3 GPTQ 4-bit",
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
            pretrained_model_name, **tokenizer_kwargs
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

        # The default GPTQ backend (EXLLAMA/HF_KERNEL) attempts to load CUDA
        # extensions that segfault on machines with TT hardware but no NVIDIA GPU.
        # Force GPTQ_TORCH, a pure-PyTorch backend that requires no CUDA.
        model_kwargs["quantization_config"] = GPTQConfig(bits=4, backend="gptq_torch")

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Dequantize GPTQ int4 weights to float tensors. GPTQ_TORCH QuantLinear
        # uses boolean-mask indexing that produces dynamic shapes incompatible with
        # XLA static-shape compilation. Replace each BaseQuantLinear with a plain
        # nn.Linear.
        from gptqmodel.nn_modules.qlinear import BaseQuantLinear

        module_map = dict(model.named_modules())
        for name, mod in list(module_map.items()):
            if not isinstance(mod, BaseQuantLinear):
                continue
            dq = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            dq.weight = nn.Parameter(mod.dequantize_weight().T.detach())
            if mod.bias is not None:
                dq.bias = nn.Parameter(mod.bias.detach())
            if dtype_override is not None:
                dq = dq.to(dtype_override)
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(module_map[parent_name], child_name, dq)
            else:
                setattr(model, name, dq)
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
