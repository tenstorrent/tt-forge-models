# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5 FP8 model loader implementation for causal language modeling.
"""
import sys
import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

# FP8 loading on CPU dequantizes to bf16, but transformers unconditionally
# imports triton (CUDA-only) at the module level in finegrained_fp8.py.
# Inject a minimal stub so the import succeeds; none of the triton kernels
# are actually invoked during CPU dequantization.
if "triton" not in sys.modules:
    _tl = types.ModuleType("triton.language")
    _tl.constexpr = type("constexpr", (), {})
    for _attr in ("float32", "float16", "bfloat16", "int8", "int32", "int64"):
        setattr(_tl, _attr, None)
    for _fn in (
        "program_id",
        "arange",
        "load",
        "store",
        "zeros",
        "max",
        "abs",
        "dot",
        "cdiv",
    ):
        setattr(_tl, _fn, lambda *a, **kw: None)

    _triton = types.ModuleType("triton")
    _triton.language = _tl
    _triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    _triton.cdiv = lambda a, b: (a + b - 1) // b
    _triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()

    sys.modules["triton"] = _triton
    sys.modules["triton.language"] = _tl

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
    """Available GLM-5 FP8 model variants for causal language modeling."""

    GLM_5_FP8 = "FP8"


class ModelLoader(ForgeModel):
    """GLM-5 FP8 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_5_FP8: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-5-FP8",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_5_FP8

    sample_text = "Give me a short introduction to large language model."

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
            model="GLM-5 FP8",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
