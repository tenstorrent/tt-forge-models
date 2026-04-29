# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Distill model loader implementation for causal language modeling.

Supports distilled variants of DeepSeek-R1 that are compatible with
HuggingFace Transformers (the full 671B MoE model is not).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available DeepSeek R1 Distill model variants."""

    DISTILL_QWEN_1_5B = "Distill_Qwen_1_5B"
    DISTILL_QWEN_1_5B_EDCASTR_JAVASCRIPT_V8 = "Distill_Qwen_1_5B_edcastr_JavaScript_v8"
    DISTILL_QWEN_7B = "Distill_Qwen_7B"
    DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT = "Distill_Qwen_7B_unsloth_bnb_4bit"
    DISTILL_QWEN_14B = "Distill_Qwen_14B"
    DISTILL_QWEN_14B_FP8_DYNAMIC = "Distill_Qwen_14B_FP8_dynamic"
    DISTILL_LLAMA_8B = "Distill_Llama_8B"
    DISTILL_LLAMA_8B_UNSLOTH = "Distill_Llama_8B_unsloth"
    DISTILL_LLAMA_70B = "Distill_Llama_70B"
    DISTILL_LLAMA_70B_BNB_4BIT = "Distill_Llama_70B_bnb_4bit"
    DEEPSEEK_R1_OPUS = "DeepSeek_R1_Opus"


class ModelLoader(ForgeModel):
    """DeepSeek R1 Distill model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DISTILL_QWEN_1_5B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_1_5B_EDCASTR_JAVASCRIPT_V8: LLMModelConfig(
            pretrained_model_name="Edcastro/DeepSeek-R1-Distill-Qwen-1.5B-edcastr_JavaScript-v8",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_7B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
            max_length=2048,
        ),
        ModelVariant.DISTILL_QWEN_14B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            max_length=2048,
        ),
        # RedHatAI FP8 dynamically quantized variant
        ModelVariant.DISTILL_QWEN_14B_FP8_DYNAMIC: LLMModelConfig(
            pretrained_model_name="RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_8B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_8B_UNSLOTH: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_70B: LLMModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            max_length=2048,
        ),
        ModelVariant.DISTILL_LLAMA_70B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
            max_length=2048,
        ),
        # squ11z1/DeepSeek-R1-Opus: safety-aligned LoRA fine-tune of
        # DeepSeek-R1-Distill-Qwen-1.5B with merged bf16 weights stored in
        # the ``bf16`` subfolder of the repository.
        ModelVariant.DEEPSEEK_R1_OPUS: LLMModelConfig(
            pretrained_model_name="squ11z1/DeepSeek-R1-Opus",
            max_length=2048,
        ),
    }

    # Per-variant subfolder where the merged model weights live when not at
    # the repository root.
    _SUBFOLDERS = {
        ModelVariant.DEEPSEEK_R1_OPUS: "bf16",
    }

    # Per-variant GGUF filename for variants that load from a .gguf checkpoint.
    _GGUF_FILES: dict = {}

    DEFAULT_VARIANT = ModelVariant.DISTILL_QWEN_1_5B

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-R1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_gguf_variant(self):
        """Check if the current variant uses GGUF quantization."""
        return self._variant in self._GGUF_FILES

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self._gguf_file
        subfolder = self._SUBFOLDERS.get(self._variant)
        if subfolder is not None:
            tokenizer_kwargs["subfolder"] = subfolder

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    @staticmethod
    def _dequantize_bnb_model(model, dtype=torch.bfloat16):
        """Replace all Linear4bit layers with dequantized regular Linear layers.

        BNB 4-bit models use Params4bit tensors that cannot be moved to TT
        device via .to(). This replaces each quantized layer with an ordinary
        nn.Linear so the model can be transferred to any device.
        """
        try:
            from bitsandbytes.nn import Linear4bit
        except ImportError:
            return model

        replacements = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, Linear4bit)
        }
        for name, module in replacements.items():
            qs = getattr(module.weight, "quant_state", None)
            if qs is not None:
                import bitsandbytes.functional as bnb_F

                weight_data = bnb_F.dequantize_4bit(module.weight.data, qs).to(dtype)
            else:
                weight_data = module.weight.data.to(dtype)

            new_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=dtype,
            )
            new_linear.weight = nn.Parameter(weight_data)
            if module.bias is not None:
                new_linear.bias = nn.Parameter(module.bias.to(dtype))

            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_linear)

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        if self._variant in (ModelVariant.DISTILL_LLAMA_70B_BNB_4BIT,):
            model_kwargs["device_map"] = "cpu"
        subfolder = self._SUBFOLDERS.get(self._variant)
        if subfolder is not None:
            model_kwargs["subfolder"] = subfolder
        model_kwargs |= kwargs

        # BNB 4-bit variants: load on CPU then dequantize to bfloat16
        # so the model can be moved to TT device (Params4bit is CUDA-only).
        if self._variant in (ModelVariant.DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT,):
            model_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self._variant in (ModelVariant.DISTILL_QWEN_7B_UNSLOTH_BNB_4BIT,):
            model = self._dequantize_bnb_model(
                model, dtype=dtype_override or torch.bfloat16
            )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
