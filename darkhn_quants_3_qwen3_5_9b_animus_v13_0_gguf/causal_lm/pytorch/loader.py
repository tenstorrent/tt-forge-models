# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Darkhn-Quants-3 Qwen3.5-9B-Animus-V13.0 GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_qwen35_support():
    """Register qwen35 as an alias for qwen3 in the GGUF loader."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "qwen35",
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"],
            )
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )


def _find_real_load_gguf_checkpoint():
    """Traverse the chain of monkey-patched load_gguf_checkpoint to find the real transformers function.

    Various loaders capture the previous function under different variable names:
    - module-level: ``from ... import load_gguf_checkpoint as _orig_load_gguf_checkpoint``
    - inside a patch function: ``orig_load = gguf_utils.load_gguf_checkpoint``

    The latter uses a closure variable, not a module global, so we must check
    both __globals__ and __closure__/__code__.co_freevars.
    """
    import importlib.util as _iutil

    spec = _iutil.find_spec("transformers.modeling_gguf_pytorch_utils")
    target_file = spec.origin if spec is not None else None

    _ORIG_NAMES = (
        "_orig_load_gguf_checkpoint",
        "orig_load",
        "_real_load_gguf_checkpoint",
        "orig_fn",
    )

    fn = _orig_load_gguf_checkpoint
    seen: set = set()
    while fn is not None:
        fid = id(fn)
        if fid in seen:
            break
        seen.add(fid)

        co = getattr(fn, "__code__", None)
        co_filename = getattr(co, "co_filename", "") if co else ""

        # Found the real transformers function (by exact file path or by absence of tt_forge_models)
        if target_file and co_filename == target_file:
            return fn
        if not target_file and "tt_forge_models" not in co_filename:
            return fn

        # Try globals (module-level ``from ... import ... as _orig_load_gguf_checkpoint``)
        next_fn = None
        for name in _ORIG_NAMES:
            v = fn.__globals__.get(name)
            if v is not None and callable(v) and id(v) != fid and id(v) not in seen:
                next_fn = v
                break

        # Try closure (inner function capturing ``orig_load = gguf_utils.load_gguf_checkpoint``)
        if next_fn is None and fn.__closure__:
            freevars = getattr(co, "co_freevars", ()) if co else ()
            for i, name in enumerate(freevars):
                if name in _ORIG_NAMES:
                    try:
                        v = fn.__closure__[i].cell_contents
                        if callable(v) and id(v) != fid and id(v) not in seen:
                            next_fn = v
                            break
                    except ValueError:
                        pass

        fn = next_fn

    # Fallback: couldn't traverse to real function; return captured orig (best effort)
    return _orig_load_gguf_checkpoint


_real_load_gguf_checkpoint = _find_real_load_gguf_checkpoint()


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_qwen35_support()
    result = _real_load_gguf_checkpoint(*args, **kwargs)
    if isinstance(result, dict) and result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3"
    return result


def _apply_patches():
    _patch_qwen35_support()
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_apply_patches()

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
    """Available Darkhn-Quants-3 Qwen3.5-9B-Animus-V13.0 GGUF model variants for causal language modeling."""

    QWEN3_5_9B_ANIMUS_V13_0_GGUF = "9B_Animus_V13.0_GGUF"


class ModelLoader(ForgeModel):
    """Darkhn-Quants-3 Qwen3.5-9B-Animus-V13.0 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_9B_ANIMUS_V13_0_GGUF: LLMModelConfig(
            pretrained_model_name="Darkhn-Quants-3/Qwen3.5-9B-Animus-V13.0-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_9B_ANIMUS_V13_0_GGUF

    GGUF_FILE = "Qwen3.5-9B-Animus-V13.0-Q4_K_M.gguf"

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
            model="Darkhn-Quants-3 Qwen3.5-9B-Animus-V13.0 GGUF",
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
        _apply_patches()
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
            enable_thinking=True,
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        _apply_patches()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
