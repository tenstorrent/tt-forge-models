# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
theprint TextSynth-8B GGUF model loader implementation for causal language modeling.

GGUF-quantized release by theprint of TextSynth-8B, a Llama 3.1 8B finetune. The
loader targets the Q4_K_M quantization.
"""
import importlib.metadata
import inspect
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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


def _prepare_gguf_env():
    """Fix two runtime GGUF issues before any transformers GGUF call.

    1. PACKAGE_DISTRIBUTION_MAPPING fix: transformers caches
       importlib.metadata.packages_distributions() at import time. If gguf is
       installed mid-session the cached map is stale, causing is_gguf_available()
       to return version 'N/A' and packaging.version.InvalidVersion. Refresh it.

    2. model_to_load fix: some model loaders patch load_gguf_checkpoint with a
       signature that omits model_to_load. Newer transformers passes that kwarg,
       raising TypeError. Wrap the live function at call time to accept it.
    """
    import transformers.configuration_utils as _config_utils
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.modeling_utils as _modeling_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.utils.import_utils as _import_utils

    # Fix 1: refresh stale PACKAGE_DISTRIBUTION_MAPPING
    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        try:
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING.update(
                importlib.metadata.packages_distributions()
            )
        except Exception:
            pass

    # Fix 2: ensure load_gguf_checkpoint accepts model_to_load.
    # Many models patch load_gguf_checkpoint with a 2-arg signature that strips
    # model_to_load; the resulting patcher chain loses it before reaching the
    # original transformers function, which then calls
    # get_gguf_hf_weights_map(None, processor) and crashes.
    # Strategy: find the *original* transformers function by traversing the
    # closure chain, then bypass the patcher chain for the tensor-loading path
    # (return_tensors=True) where model_to_load is required.

    def _find_original_load_gguf(fn, _seen=None):
        if _seen is None:
            _seen = set()
        fn_id = id(fn)
        if fn_id in _seen:
            return None
        _seen.add(fn_id)
        if (
            getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
            and getattr(fn, "__qualname__", "") == "load_gguf_checkpoint"
        ):
            return fn
        try:
            _cvars = inspect.getclosurevars(fn)
            # nonlocals: closure vars (functions defined inside other functions)
            # globals: module-level vars referenced by the function (module-level patchers)
            _candidates = dict(_cvars.nonlocals)
            _candidates.update(_cvars.globals)
        except Exception:
            return None
        for _val in _candidates.values():
            if callable(_val) and _val is not fn:
                _r = _find_original_load_gguf(_val, _seen)
                if _r is not None:
                    return _r
        return None

    _current = _gguf_utils.load_gguf_checkpoint
    if "model_to_load" not in inspect.signature(_current).parameters:
        _captured = _current
        _original = _find_original_load_gguf(_current)
        if (
            _original is not None
            and "model_to_load" in inspect.signature(_original).parameters
        ):

            def _wrapped(gguf_path, return_tensors=False, model_to_load=None):
                if return_tensors and model_to_load is not None:
                    # Call original directly; the patcher chain strips model_to_load
                    return _original(
                        gguf_path,
                        return_tensors=True,
                        model_to_load=model_to_load,
                    )
                return _captured(gguf_path, return_tensors=return_tensors)

        else:
            # Could not find original; fall back to stripping model_to_load
            def _wrapped(gguf_path, return_tensors=False, model_to_load=None):
                return _captured(gguf_path, return_tensors=return_tensors)

        for _mod in (_gguf_utils, _config_utils, _auto_tokenizer, _modeling_utils):
            _mod.load_gguf_checkpoint = _wrapped


class ModelVariant(StrEnum):
    """Available theprint TextSynth-8B GGUF model variants for causal language modeling."""

    TEXTSYNTH_8B_Q4_K_M_GGUF = "TextSynth_8B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """theprint TextSynth-8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TEXTSYNTH_8B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="theprint/TextSynth-8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEXTSYNTH_8B_Q4_K_M_GGUF

    GGUF_FILE = "TextSynth-8B.Q4_K_M.gguf"

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
            model="theprint TextSynth-8B GGUF",
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
        _prepare_gguf_env()
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        _prepare_gguf_env()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
