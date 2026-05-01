# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RichardErkhov Llama-3-8B-Magpie-Align-SFT-v0.1 GGUF model loader implementation for causal language modeling.
"""
import importlib.util
import inspect
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

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


def _find_real_load_gguf_checkpoint():
    """Traverse the patch chain to find the original transformers load_gguf_checkpoint.

    Other loaders in the pytest session may have replaced load_gguf_checkpoint
    with a narrow-sig wrapper that doesn't accept model_to_load (transformers 5.x).
    Walk the chain to recover the real function.
    """
    spec = importlib.util.find_spec("transformers.modeling_gguf_pytorch_utils")
    if spec is None or spec.origin is None:
        return None
    real_file = os.path.abspath(spec.origin)

    fn = _gguf_utils.load_gguf_checkpoint
    visited = set()

    while fn is not None:
        fn_id = id(fn)
        if fn_id in visited:
            break
        visited.add(fn_id)

        try:
            fn_file = os.path.abspath(inspect.getfile(fn))
            if fn_file == real_file:
                return fn
        except (TypeError, OSError):
            pass

        saved = getattr(fn, "_magpie_real_fn", None)
        if saved is not None and callable(saved):
            return saved

        next_fn = None
        try:
            g = fn.__globals__
            for name in ("_orig_load_gguf_checkpoint", "orig_load", "_orig"):
                candidate = g.get(name)
                if callable(candidate) and id(candidate) not in visited:
                    next_fn = candidate
                    break
        except AttributeError:
            pass

        if next_fn is None:
            try:
                freevars = getattr(fn, "__code__", None)
                freevars = freevars.co_freevars if freevars else ()
                cells = fn.__closure__ or ()
                for i, name in enumerate(freevars):
                    if name in ("orig_load", "_orig", "_orig_load_gguf_checkpoint") and i < len(cells):
                        try:
                            val = cells[i].cell_contents
                            if callable(val) and id(val) not in visited:
                                next_fn = val
                                break
                        except ValueError:
                            pass
            except AttributeError:
                pass

        fn = next_fn

    return None


def _ensure_gguf_patch():
    """Install a wide-sig load_gguf_checkpoint that correctly forwards model_to_load.

    Called at import time and again in load_model() to win any ordering race
    with other loaders that install narrow-sig wrappers.
    """
    real_fn = _find_real_load_gguf_checkpoint()
    if real_fn is None:
        return

    current = _gguf_utils.load_gguf_checkpoint
    if (
        getattr(current, "__name__", "") == "_patched_load_gguf_checkpoint"
        and getattr(current, "_magpie_real_fn", None) is real_fn
    ):
        return

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        return real_fn(*args, **kwargs)

    _patched_load_gguf_checkpoint.__name__ = "_patched_load_gguf_checkpoint"
    _patched_load_gguf_checkpoint._magpie_real_fn = real_fn

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_ensure_gguf_patch()


class ModelVariant(StrEnum):
    """Available Llama-3-8B-Magpie-Align-SFT-v0.1 GGUF model variants for causal language modeling."""

    LLAMA_3_8B_MAGPIE_ALIGN_SFT_V0_1_GGUF = "8B_Magpie_Align_SFT_v0_1_GGUF"


class ModelLoader(ForgeModel):
    """Llama-3-8B-Magpie-Align-SFT-v0.1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_8B_MAGPIE_ALIGN_SFT_V0_1_GGUF: LLMModelConfig(
            pretrained_model_name="RichardErkhov/Magpie-Align_-_Llama-3-8B-Magpie-Align-SFT-v0.1-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_8B_MAGPIE_ALIGN_SFT_V0_1_GGUF

    GGUF_FILE = "Llama-3-8B-Magpie-Align-SFT-v0.1.Q4_K_M.gguf"

    sample_text = "What is your favorite city?"

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
            model="Llama-3-8B-Magpie-Align-SFT-v0.1 GGUF",
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
        _ensure_gguf_patch()
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
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
