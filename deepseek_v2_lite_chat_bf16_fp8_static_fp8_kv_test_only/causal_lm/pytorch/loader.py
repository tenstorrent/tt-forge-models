# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V2-Lite-Chat BF16 FP8-STATIC FP8-KV model loader implementation for causal language modeling.

Loads INC4AI's compressed-tensors quantized variant of deepseek-ai/DeepSeek-V2-Lite-Chat,
which applies BF16 weights with FP8 static activation quantization and FP8 KV cache.
"""
import contextlib
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


@contextlib.contextmanager
def _skip_fp8_weight_init():
    """Prevent normal_() from failing on Float8 tensors during weight initialization."""
    original_normal_ = torch.Tensor.normal_

    def patched_normal_(self, mean=0, std=1):
        if self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return self
        return original_normal_(self, mean=mean, std=std)

    torch.Tensor.normal_ = patched_normal_
    try:
        yield
    finally:
        torch.Tensor.normal_ = original_normal_


class ModelVariant(StrEnum):
    """Available DeepSeek-V2-Lite-Chat BF16 FP8-STATIC FP8-KV model variants for causal language modeling."""

    DEEPSEEK_V2_LITE_CHAT_BF16_FP8_STATIC_FP8_KV_TEST_ONLY = (
        "DeepSeek_V2_Lite_Chat_BF16_FP8_STATIC_FP8_KV_TEST_ONLY"
    )


class ModelLoader(ForgeModel):
    """DeepSeek-V2-Lite-Chat BF16 FP8-STATIC FP8-KV model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_V2_LITE_CHAT_BF16_FP8_STATIC_FP8_KV_TEST_ONLY: LLMModelConfig(
            pretrained_model_name="INC4AI/DeepSeek-V2-Lite-Chat-BF16-FP8-STATIC-FP8-KV-TEST-ONLY",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.DEEPSEEK_V2_LITE_CHAT_BF16_FP8_STATIC_FP8_KV_TEST_ONLY
    )

    sample_text = "Give me a short introduction to large language models."

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
            model="DeepSeek-V2-Lite-Chat BF16 FP8-STATIC FP8-KV",
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _skip_fp8_weight_init():
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
