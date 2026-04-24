# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Blossom-V6-7B i1 GGUF model loader implementation for causal language modeling.
"""
import inspect
from contextlib import contextmanager
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def _find_original_gguf_loader(fn):
    """Walk the patch chain (closures and module globals) to find the actual transformers load_gguf_checkpoint."""
    visited = set()
    candidates = [fn]
    while candidates:
        current = candidates.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        try:
            sig = inspect.signature(current)
            if "model_to_load" in sig.parameters:
                return current
        except Exception:
            pass
        # Module-level patches store the wrapped function in module globals under _orig_* names
        if hasattr(current, "__globals__"):
            for key in ("_orig_load_gguf_checkpoint", "_original_load_gguf_checkpoint"):
                val = current.__globals__.get(key)
                if callable(val) and id(val) not in visited:
                    candidates.append(val)
        # Closure-based patches
        if hasattr(current, "__closure__") and current.__closure__:
            for cell in current.__closure__:
                try:
                    content = cell.cell_contents
                    if callable(content):
                        candidates.append(content)
                except ValueError:
                    pass
    return None


# Capture the original at import time; works whether we're imported before or after qwen35 loaders
_orig_gguf_loader = _find_original_gguf_loader(_gguf_utils.load_gguf_checkpoint)


@contextmanager
def _original_gguf_loader_ctx():
    """Temporarily restore the original load_gguf_checkpoint for non-qwen35 GGUF models.

    qwen35 loaders patch load_gguf_checkpoint globally.  Since pytest collects all
    loaders before running any test, the patch is always active at call time even if
    this module was imported first.  We save the current (patched) references, swap
    in the originals, yield, and then restore – so other tests are unaffected.
    """
    if _orig_gguf_loader is None:
        yield
        return
    saved_gguf = _gguf_utils.load_gguf_checkpoint
    saved_cfg = _config_utils.load_gguf_checkpoint
    saved_tok_auto = _auto_tokenizer.load_gguf_checkpoint
    saved_tok = _tok_utils.load_gguf_checkpoint
    _gguf_utils.load_gguf_checkpoint = _orig_gguf_loader
    _config_utils.load_gguf_checkpoint = _orig_gguf_loader
    _auto_tokenizer.load_gguf_checkpoint = _orig_gguf_loader
    _tok_utils.load_gguf_checkpoint = _orig_gguf_loader
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = saved_gguf
        _config_utils.load_gguf_checkpoint = saved_cfg
        _auto_tokenizer.load_gguf_checkpoint = saved_tok_auto
        _tok_utils.load_gguf_checkpoint = saved_tok


class ModelVariant(StrEnum):
    """Available mradermacher Blossom-V6-7B i1 GGUF model variants for causal language modeling."""

    MRADERMACHER_BLOSSOM_V6_7B_I1_GGUF = "V6_7B_i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Blossom-V6-7B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MRADERMACHER_BLOSSOM_V6_7B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Blossom-V6-7B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRADERMACHER_BLOSSOM_V6_7B_I1_GGUF

    GGUF_FILE = "Blossom-V6-7B.i1-Q4_K_M.gguf"

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
            model="mradermacher Blossom-V6-7B i1 GGUF",
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

        with _original_gguf_loader_ctx():
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
            with _original_gguf_loader_ctx():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _original_gguf_loader_ctx():
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _original_gguf_loader_ctx():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
