# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama-3 SQLCoder 8B GGUF model loader implementation for causal language modeling.
"""
import functools
import inspect
import threading
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_base as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

_gguf_ctx = threading.local()


def _patch_gguf_compat():
    def _wrap_load(fn):
        try:
            sig = inspect.signature(fn)
            if "model_to_load" in sig.parameters and not getattr(
                fn, "_gguf_compat_load_wrapped", False
            ):
                return fn
        except (ValueError, TypeError):
            pass
        if getattr(fn, "_gguf_compat_load_wrapped", False):
            return fn

        @functools.wraps(fn)
        def _wrapper(gguf_path, return_tensors=False, model_to_load=None, **kwargs):
            _gguf_ctx.model = model_to_load
            try:
                return fn(gguf_path, return_tensors=return_tensors)
            finally:
                _gguf_ctx.model = None

        _wrapper._gguf_compat_load_wrapped = True
        return _wrapper

    def _wrap_get_map(fn):
        if getattr(fn, "_gguf_compat_map_wrapped", False):
            return fn

        @functools.wraps(fn)
        def _wrapper(hf_model, *args, **kwargs):
            if hf_model is None:
                hf_model = getattr(_gguf_ctx, "model", None)
            return fn(hf_model, *args, **kwargs)

        _wrapper._gguf_compat_map_wrapped = True
        return _wrapper

    if hasattr(_gguf_utils, "get_gguf_hf_weights_map"):
        _gguf_utils.get_gguf_hf_weights_map = _wrap_get_map(
            _gguf_utils.get_gguf_hf_weights_map
        )

    for _mod in (_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _wrap_load(_mod.load_gguf_checkpoint)


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
    """Available Llama-3 SQLCoder 8B GGUF model variants for causal language modeling."""

    LLAMA_3_SQLCODER_8B_GGUF = "8B_GGUF"


class ModelLoader(ForgeModel):
    """Llama-3 SQLCoder 8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_SQLCODER_8B_GGUF: LLMModelConfig(
            pretrained_model_name="RichardErkhov/defog_-_llama-3-sqlcoder-8b-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_SQLCODER_8B_GGUF

    GGUF_FILE = "llama-3-sqlcoder-8b.Q4_K_M.gguf"

    sample_text = "SELECT * FROM users WHERE"

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
            model="Llama-3 SQLCoder 8B GGUF",
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

        _patch_gguf_compat()
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

        prompts = [self.sample_text]

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
        _patch_gguf_compat()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
