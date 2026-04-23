# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model loader implementation for causal language modeling.
"""
import contextlib
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache/hub"
import huggingface_hub.constants as _hf_constants

_hf_constants.HF_HOME = "/tmp/hf_cache"
_hf_constants.HF_HUB_CACHE = "/tmp/hf_cache/hub"

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


def _find_true_original_load_gguf(fn):
    """Traverse the monkey-patch chain to find the original transformers function.

    Other GGUF loaders wrap load_gguf_checkpoint using different patterns:
    - Some store the previous version as _orig_load_gguf_checkpoint in module globals
    - Some store it as orig_load in a closure variable
    We follow the chain until we reach a function from the transformers package itself.
    """
    visited = set()
    current = fn
    while True:
        fn_id = id(current)
        if fn_id in visited:
            break
        visited.add(fn_id)
        module = getattr(current, "__module__", "") or ""
        if module.startswith("transformers."):
            return current
        advanced = False
        globals_ = getattr(current, "__globals__", {})
        prev = globals_.get("_orig_load_gguf_checkpoint")
        if prev is not None:
            current = prev
            advanced = True
            continue
        closure = getattr(current, "__closure__", None) or ()
        code = getattr(current, "__code__", None)
        freevars = getattr(code, "co_freevars", ()) if code else ()
        for i, name in enumerate(freevars):
            if name in ("orig_load", "_orig_load", "_orig_load_gguf_checkpoint"):
                if i < len(closure):
                    try:
                        cell_contents = closure[i].cell_contents
                        if callable(cell_contents):
                            current = cell_contents
                            advanced = True
                            break
                    except ValueError:
                        pass
        if not advanced:
            break
    return current


@contextlib.contextmanager
def _compat_gguf_ctx():
    """Temporarily patch load_gguf_checkpoint to restore model_to_load support.

    Other GGUF loaders monkey-patch load_gguf_checkpoint without forwarding
    the model_to_load parameter added in transformers 5.2.0.  We replace the
    patched chain with a thin wrapper around the real transformers function
    that passes model_to_load correctly.  The wrapper is installed only while
    a from_pretrained call is in flight and restored afterward.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf

    _patched_chain = _gguf.load_gguf_checkpoint
    _true_original = _find_true_original_load_gguf(_patched_chain)

    def _compat(gguf_path, return_tensors=False, model_to_load=None, **kw):
        return _true_original(
            gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
        )

    _gguf.load_gguf_checkpoint = _compat
    try:
        yield
    finally:
        _gguf.load_gguf_checkpoint = _patched_chain


class ModelVariant(StrEnum):
    """Available Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model variants for causal language modeling."""

    QWEN3_7B_INSTRUCT_Q4_K_M_GGUF = "Qwen3_7B_Instruct_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_7B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Ygz-08123/Qwen3-7B-Instruct-Q4_K_M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_7B_INSTRUCT_Q4_K_M_GGUF

    GGUF_FILE = "qwen3-7b-instruct-q4_k_m.gguf"

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
            model="Ygz-08123 Qwen3-7B-Instruct Q4_K_M GGUF",
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

        with _compat_gguf_ctx():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                cache_dir="/tmp/hf_cache",
                **tokenizer_kwargs,
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
            with _compat_gguf_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name,
                    gguf_file=self.GGUF_FILE,
                    cache_dir="/tmp/hf_cache",
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _compat_gguf_ctx():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, cache_dir="/tmp/hf_cache", **model_kwargs
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
        with _compat_gguf_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                cache_dir="/tmp/hf_cache",
            )
        return self.config
