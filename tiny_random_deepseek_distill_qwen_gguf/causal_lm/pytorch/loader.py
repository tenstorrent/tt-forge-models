# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random DeepSeek Distill Qwen GGUF model loader implementation for causal language modeling.
"""
import inspect
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _find_gguf_load_with_model_to_load(fn):
    """Search closure chain and module globals for load_gguf_checkpoint accepting model_to_load.

    Other loaders patch load_gguf_checkpoint at module level without forwarding
    model_to_load (required by transformers 5.x). Walk the patch chain to find a
    version that accepts it.
    """
    seen = set()
    queue = [fn]
    while queue:
        f = queue.pop(0)
        fid = id(f)
        if fid in seen:
            continue
        seen.add(fid)
        try:
            sig = inspect.signature(f)
            if "model_to_load" in sig.parameters:
                return f
        except (ValueError, TypeError):
            pass
        if hasattr(f, "__closure__") and f.__closure__:
            for cell in f.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        queue.append(val)
                except ValueError:
                    pass
        if hasattr(f, "__globals__"):
            for name, val in f.__globals__.items():
                if callable(val) and "load_gguf" in name and id(val) not in seen:
                    queue.append(val)
    return None


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
    """Available Tiny Random DeepSeek Distill Qwen GGUF model variants for causal language modeling."""

    TINY_RANDOM_DEEPSEEK_DISTILL_QWEN_Q8_0 = "tiny_random_deepseek_distill_qwen_q8_0"


class ModelLoader(ForgeModel):
    """Tiny Random DeepSeek Distill Qwen GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_DEEPSEEK_DISTILL_QWEN_Q8_0: LLMModelConfig(
            pretrained_model_name="sammysun0711/tiny-random-deepseek-distill-qwen-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_DEEPSEEK_DISTILL_QWEN_Q8_0

    GGUF_FILE = "tiny-random-deepseek-distill-qwen_q8_0.gguf"

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

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
            model="Tiny Random DeepSeek Distill Qwen GGUF",
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

        _current_gguf = _gguf_utils.load_gguf_checkpoint
        _orig_gguf = _find_gguf_load_with_model_to_load(_current_gguf)
        if _orig_gguf and _orig_gguf is not _current_gguf:
            _gguf_utils.load_gguf_checkpoint = _orig_gguf
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _current_gguf

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
