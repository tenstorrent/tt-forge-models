# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-8B Jailbroken GGUF model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _get_real_load_gguf_checkpoint():
    """Walk the monkey-patch closure chain to recover the original load_gguf_checkpoint.

    Other GGUF model loaders patch load_gguf_checkpoint with signatures that drop
    the model_to_load kwarg added in transformers 5.2.0. This finds the real function.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    seen = set()
    while True:
        if id(fn) in seen:
            break
        seen.add(id(fn))
        if (
            getattr(fn, "__module__", None)
            == "transformers.modeling_gguf_pytorch_utils"
        ):
            return fn
        closure = fn.__closure__ or ()
        next_fn = None
        for var, cell in zip(getattr(fn.__code__, "co_freevars", ()), closure):
            if "orig" in var or "load_gguf" in var:
                try:
                    val = cell.cell_contents
                    if callable(val):
                        next_fn = val
                        break
                except ValueError:
                    pass
        if next_fn is None or next_fn is fn:
            break
        fn = next_fn
    return fn


def _compat_load_gguf_checkpoint(*args, **kwargs):
    """Forward all kwargs (including model_to_load) to the real load_gguf_checkpoint."""
    return _get_real_load_gguf_checkpoint()(*args, **kwargs)


_gguf_utils.load_gguf_checkpoint = _compat_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _compat_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _compat_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _compat_load_gguf_checkpoint

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
    """Available Qwen3-8B Jailbroken GGUF model variants for causal language modeling."""

    QWEN3_8B_JAILBROKEN_GGUF = "8B_Jailbroken_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3-8B Jailbroken GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_8B_JAILBROKEN_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-8B-Jailbroken-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_8B_JAILBROKEN_GGUF

    GGUF_FILE = "Qwen3-8B-Jailbroken.Q4_K_M.gguf"

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
            model="Qwen3-8B Jailbroken GGUF",
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

        messages = [{"role": "user", "content": self.sample_text}]
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
