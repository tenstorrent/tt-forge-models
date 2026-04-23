# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 3 4B IT Heretic Creative GGUF model loader implementation for causal language modeling.
"""
import functools
import inspect
import threading
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_base as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

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

# Thread-local storage so model_to_load survives the monkey-patch chain that
# strips it from intermediate _patched_load_gguf_checkpoint signatures.
_gguf_ctx = threading.local()


def _patch_gguf_compat():
    """Re-wrap load_gguf_checkpoint / get_gguf_hf_weights_map for transformers 5.2.0.

    transformers 5.2.0 added a model_to_load parameter that is required for
    get_gguf_hf_weights_map (state_dict lookup).  Other GGUF loaders
    monkey-patch load_gguf_checkpoint with fixed (gguf_path, return_tensors)
    signatures that drop model_to_load.  We:
      1. Wrap load_gguf_checkpoint to accept model_to_load and stash it in a
         thread-local before delegating to the existing patch chain.
      2. Wrap get_gguf_hf_weights_map to recover model_to_load from the
         thread-local when the chain eventually passes None.
    Both wrappers are applied at call time so they are always the outermost
    layer, regardless of which loaders were collected by pytest.
    """

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


class ModelVariant(StrEnum):
    """Available Gemma 3 4B IT Heretic Creative GGUF model variants for causal language modeling."""

    GEMMA_3_4B_IT_HERETIC_CREATIVE_GGUF = "4B_IT_HERETIC_CREATIVE_GGUF"


class ModelLoader(ForgeModel):
    """Gemma 3 4B IT Heretic Creative GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT_HERETIC_CREATIVE_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/gemma-3-4b-it-heretic-creative-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_4B_IT_HERETIC_CREATIVE_GGUF

    GGUF_FILE = "gemma-3-4b-it-heretic-creative.Q4_K_M.gguf"

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
            model="Gemma 3 4B IT Heretic Creative GGUF",
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
        _patch_gguf_compat()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
