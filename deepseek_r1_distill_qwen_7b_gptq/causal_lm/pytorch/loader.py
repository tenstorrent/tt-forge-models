# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek R1 Distill Qwen 7B GPTQ model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# gptqmodel >= 3.0 removed BACKEND.EXLLAMA_V1; patch optimum's module-level
# BACKEND reference so the post_init_model comparison doesn't raise AttributeError.
try:
    from gptqmodel import BACKEND as _GPTQ_BACKEND

    if not hasattr(_GPTQ_BACKEND, "EXLLAMA_V1"):
        import optimum.gptq.quantizer as _opt_quant

        class _BackendShim:
            """Proxy that forwards attribute lookups to the real BACKEND but adds EXLLAMA_V1."""

            EXLLAMA_V1 = "exllama_v1_removed"

            def __getattr__(self, name):
                return getattr(_GPTQ_BACKEND, name)

        _opt_quant.BACKEND = _BackendShim()
except Exception:
    pass

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
    """Available DeepSeek R1 Distill Qwen 7B GPTQ model variants."""

    DISTILL_QWEN_7B_GPTQ_INT4 = "Distill_Qwen_7B_GPTQ_Int4"


class ModelLoader(ForgeModel):
    """DeepSeek R1 Distill Qwen 7B GPTQ model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.DISTILL_QWEN_7B_GPTQ_INT4: LLMModelConfig(
            pretrained_model_name="AngelSlim/Deepseek_r1_distill_qwen-7b_int4_gptq",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTILL_QWEN_7B_GPTQ_INT4

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek R1 Distill Qwen 7B GPTQ",
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
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {
            "device_map": "cpu",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
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
