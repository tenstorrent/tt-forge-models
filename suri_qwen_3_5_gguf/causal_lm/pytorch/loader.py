# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Suri Qwen 3.5 GGUF model loader implementation for causal language modeling.
"""
import threading
import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

_tls = threading.local()


def _fix_gguf_patch():
    """Re-apply compat shims immediately before from_pretrained calls.

    Many GGUF loaders patch load_gguf_checkpoint without accepting the
    model_to_load kwarg added in newer transformers, stripping it from
    the call chain.  We save model_to_load in thread-local storage before
    the broken chain drops it, then restore it in a companion patch on
    get_gguf_hf_weights_map so that loaders patching that function receive
    a valid model instead of None.
    """
    if getattr(_gguf_utils.load_gguf_checkpoint, "_model_to_load_compat", False):
        return
    _current_load = _gguf_utils.load_gguf_checkpoint
    _current_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _compat_load(*args, **kwargs):
        _tls.model_to_load = kwargs.pop("model_to_load", None)
        try:
            return _current_load(*args, **kwargs)
        finally:
            _tls.model_to_load = None

    def _compat_get_map(hf_model, *args, **kwargs):
        if hf_model is None:
            hf_model = getattr(_tls, "model_to_load", None)
        return _current_get_map(hf_model, *args, **kwargs)

    _compat_load._model_to_load_compat = True
    _gguf_utils.load_gguf_checkpoint = _compat_load
    _gguf_utils.get_gguf_hf_weights_map = _compat_get_map


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
    """Available Suri Qwen 3.5 GGUF model variants for causal language modeling."""

    SURI_QWEN_3_5_4B_UNCENSORED_I1_Q4_K_M = "4B_Uncensored_i1_Q4_K_M"
    SURI_QWEN_3_5_4B_UNCENSORED_Q4_K_M = "4B_Uncensored_Q4_K_M"


class ModelLoader(ForgeModel):
    """Suri Qwen 3.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SURI_QWEN_3_5_4B_UNCENSORED_I1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Suri-Qwen-3.5-4B-Uncensored-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.SURI_QWEN_3_5_4B_UNCENSORED_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/Suri-Qwen-3.5-4B-Uncensored-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SURI_QWEN_3_5_4B_UNCENSORED_I1_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.SURI_QWEN_3_5_4B_UNCENSORED_I1_Q4_K_M: "Suri-Qwen-3.5-4B-Uncensored.i1-Q4_K_M.gguf",
        ModelVariant.SURI_QWEN_3_5_4B_UNCENSORED_Q4_K_M: "Suri-Qwen-3.5-4B-Uncensored.Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Suri Qwen 3.5 GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

        _fix_gguf_patch()
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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        _fix_gguf_patch()
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
        _fix_gguf_patch()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
