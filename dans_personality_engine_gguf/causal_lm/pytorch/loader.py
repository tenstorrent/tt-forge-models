# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dans PersonalityEngine GGUF model loader implementation for causal language modeling.
"""
import inspect

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tok
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _unwrap_to_real_load_gguf_checkpoint():
    """Walk the monkey-patch chain (closures + module globals) to find the real transformers function.

    Other loaders patch load_gguf_checkpoint at import time, forming a chain
    where each wrapper drops model_to_load. We need the real transformers
    function which accepts model_to_load so GGUF tensor mapping works correctly.
    Some patches store the original via closure variables; others via module-level
    globals (co_names). We check both.
    """
    fn = _gguf_utils.load_gguf_checkpoint
    visited = set()
    while fn is not None and id(fn) not in visited:
        visited.add(id(fn))
        try:
            if "model_to_load" in inspect.signature(fn).parameters:
                return fn
        except (TypeError, ValueError):
            pass
        next_fn = None
        # Check closure variables
        if hasattr(fn, "__code__") and fn.__closure__ and fn.__code__.co_freevars:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if not callable(val) or id(val) in visited:
                    continue
                if any(k in name.lower() for k in ("orig", "load")):
                    next_fn = val
                    break
                if next_fn is None:
                    next_fn = val
        # Also check module-level globals referenced by name (patches that use globals
        # instead of closures to store the original function)
        if next_fn is None and hasattr(fn, "__globals__") and hasattr(fn, "__code__"):
            for name in fn.__code__.co_names:
                if not any(k in name.lower() for k in ("orig", "load")):
                    continue
                val = fn.__globals__.get(name)
                if val is None or not callable(val) or id(val) in visited:
                    continue
                next_fn = val
                break
        if next_fn is None:
            break
        fn = next_fn
    return None


def _apply_model_to_load_compat():
    real_fn = _unwrap_to_real_load_gguf_checkpoint()
    if real_fn is not None:
        _gguf_utils.load_gguf_checkpoint = real_fn
        for _mod in (_config_utils, _auto_tok, _tok_utils):
            _mod.load_gguf_checkpoint = real_fn
    else:
        # Fallback: wrap to at least accept model_to_load kwarg
        fn = _gguf_utils.load_gguf_checkpoint
        if "model_to_load" not in inspect.signature(fn).parameters:

            def _wrapper(gguf_path, return_tensors=False, model_to_load=None):
                return fn(gguf_path, return_tensors=return_tensors)

            _gguf_utils.load_gguf_checkpoint = _wrapper
            for _mod in (_config_utils, _auto_tok, _tok_utils):
                _mod.load_gguf_checkpoint = _wrapper


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
    """Available Dans PersonalityEngine GGUF model variants for causal language modeling."""

    DANS_PERSONALITY_ENGINE_V1_3_0_24B_GGUF = "V1_3_0_24B_GGUF"
    DANS_PERSONALITY_ENGINE_V1_3_0_12B_GGUF = "V1_3_0_12B_GGUF"


class ModelLoader(ForgeModel):
    """Dans PersonalityEngine GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_24B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/PocketDoc_Dans-PersonalityEngine-V1.3.0-24b-GGUF",
            max_length=128,
        ),
        ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_12B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/PocketDoc_Dans-PersonalityEngine-V1.3.0-12b-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_24B_GGUF

    _GGUF_FILES = {
        ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_24B_GGUF: "PocketDoc_Dans-PersonalityEngine-V1.3.0-24b-Q4_K_M.gguf",
        ModelVariant.DANS_PERSONALITY_ENGINE_V1_3_0_12B_GGUF: "PocketDoc_Dans-PersonalityEngine-V1.3.0-12b-Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

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
            model="Dans PersonalityEngine GGUF",
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

        _apply_model_to_load_compat()
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
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
