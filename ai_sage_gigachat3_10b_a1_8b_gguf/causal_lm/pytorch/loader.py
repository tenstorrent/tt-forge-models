# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ai-sage GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling.
"""
import importlib.util
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFLlamaConverter

_orig_is_gguf_available = _gguf_utils.is_gguf_available


def _patched_is_gguf_available(*args, **kwargs):
    if importlib.util.find_spec("gguf") is None:
        return False
    try:
        return _orig_is_gguf_available(*args, **kwargs)
    except Exception:
        return True


_gguf_utils.is_gguf_available = _patched_is_gguf_available

# deepseek_v2 tokenizer architecture is not in GGUF_TO_FAST_CONVERTERS; it uses
# the same sentencepiece-based tokenizer structure as llama.
GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFLlamaConverter)

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
    """Available ai-sage GigaChat3 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT3_10B_A1_8B_GGUF = "10B_A1_8B_GGUF"


class ModelLoader(ForgeModel):
    """ai-sage GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT3_10B_A1_8B_GGUF: LLMModelConfig(
            pretrained_model_name="ai-sage/GigaChat3-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT3_10B_A1_8B_GGUF

    GGUF_FILE = "GigaChat3-10B-A1.8B-q6_k.gguf"

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
            model="ai-sage GigaChat3 10B A1.8B GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
