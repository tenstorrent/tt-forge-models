# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedQwen 1.5B i1 GGUF model loader implementation for causal language modeling.
"""

import importlib.metadata
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


def _find_real_load_gguf_checkpoint(fn):
    """Traverse a patch chain to find the real load_gguf_checkpoint from transformers.

    Other GGUF loaders wrap load_gguf_checkpoint without forwarding all kwargs.
    This traverses globals and closure vars to reach the real function.
    """
    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if fn.__name__ == "load_gguf_checkpoint" and getattr(
            fn, "__module__", ""
        ).endswith("modeling_gguf_pytorch_utils"):
            return fn
        next_fn = None
        globals_dict = getattr(fn, "__globals__", {})
        for var_name in (
            "_orig_load_gguf_checkpoint",
            "orig_load",
            "_inner",
            "_current",
            "_orig",
        ):
            candidate = globals_dict.get(var_name)
            if callable(candidate) and candidate is not fn:
                next_fn = candidate
                break
        if next_fn is None:
            closure = getattr(fn, "__closure__", None) or ()
            freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
            for name, cell in zip(freevars, closure):
                try:
                    val = cell.cell_contents
                    if callable(val) and val is not fn:
                        next_fn = val
                        break
                except ValueError:
                    pass
        if next_fn is None:
            break
        fn = next_fn
    return fn


def _fix_gguf_version_detection():
    """Fix gguf version detection when installed at runtime by RequirementsManager.

    transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
    is installed later, the mapping is stale and version detection falls back to
    gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
    """
    import transformers.utils.import_utils as _import_utils

    if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        try:
            importlib.metadata.version("gguf")
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
            _import_utils.is_gguf_available.cache_clear()
        except importlib.metadata.PackageNotFoundError:
            pass


class ModelVariant(StrEnum):
    """Available MedQwen 1.5B i1 GGUF model variants for causal language modeling."""

    MEDQWEN_1_5B_I1_GGUF = "1.5B_i1_GGUF"


class ModelLoader(ForgeModel):
    """MedQwen 1.5B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MEDQWEN_1_5B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/medqwen-1.5b-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDQWEN_1_5B_I1_GGUF

    GGUF_FILE = "medqwen-1.5b.i1-Q4_K_M.gguf"

    sample_text = "What are the common symptoms of type 2 diabetes?"

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
            model="MedQwen 1.5B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _fix_gguf_version_detection()
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

        # Other GGUF loaders patch load_gguf_checkpoint without forwarding model_to_load,
        # which was added in newer transformers. Temporarily install our wrapper as the
        # outermost patch so model_to_load reaches the real function via chain traversal.
        _mods = (_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils)
        _prev = {mod: mod.load_gguf_checkpoint for mod in _mods}
        _real_fn = _find_real_load_gguf_checkpoint(_prev[_gguf_utils])

        def _wrapper(gguf_path, return_tensors=False, **patch_kwargs):
            return _real_fn(gguf_path, return_tensors=return_tensors, **patch_kwargs)

        for mod in _mods:
            mod.load_gguf_checkpoint = _wrapper
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            for mod, fn in _prev.items():
                mod.load_gguf_checkpoint = fn

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
        _fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
