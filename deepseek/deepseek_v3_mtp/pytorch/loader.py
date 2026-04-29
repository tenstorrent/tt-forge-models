# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3 MTP model loader implementation for causal language modeling.

Supports the DeepSeek V3 architecture with Multi-Token Prediction (MTP),
using a small random-weight variant for testing.
"""

import sys
import types
from typing import Optional

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


def _stub_triton_if_missing():
    # transformers finegrained_fp8.py imports triton at module level. On non-CUDA
    # hardware the quantizer sets dequantize=True so actual triton kernels are never
    # called, but the import must succeed. This stub satisfies the import without
    # requiring the NVIDIA-only triton package.
    if "triton" not in sys.modules:
        tl = types.ModuleType("triton.language")
        tl.constexpr = None  # used as function annotation; any Python object is valid
        triton_mod = types.ModuleType("triton")
        triton_mod.jit = lambda fn: fn
        triton_mod.language = tl
        sys.modules["triton"] = triton_mod
        sys.modules["triton.language"] = tl


class ModelVariant(StrEnum):
    """Available DeepSeek V3 MTP model variants."""

    MTP_DRAFT_RANDOM = "MTP_Draft_Random"
    MTP_MAIN_RANDOM = "MTP_Main_Random"


class ModelLoader(ForgeModel):
    """DeepSeek V3 MTP model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.MTP_DRAFT_RANDOM: LLMModelConfig(
            pretrained_model_name="luccafong/deepseek_mtp_draft_random",
            max_length=2048,
        ),
        ModelVariant.MTP_MAIN_RANDOM: LLMModelConfig(
            pretrained_model_name="luccafong/deepseek_mtp_main_random",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MTP_MAIN_RANDOM

    sample_text = "What is the capital of France?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-V3-MTP",
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
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _stub_triton_if_missing()
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
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
