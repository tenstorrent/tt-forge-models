# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gaiasky Qwen 3.5 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import inspect as _inspect
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

# Mutable holder — updated in _apply_gguf_patches() each time we (re-)install
# our wrapper, so the actual call always reaches the real transformers function
# regardless of how many narrow-sig patches other loaders layer on top.
_real_load_gguf_fn = [None]


def _patch_qwen35_support():
    """Register qwen35 architecture as an alias for qwen3.

    Qwen 3.5 uses the same model architecture as Qwen 3 but the GGUF file
    declares architecture as 'qwen35' which transformers 5.x does not yet
    recognise.
    """
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


def _find_real_load_gguf(fn, _seen=None):
    """Walk the patch chain to find the real load_gguf_checkpoint.

    Each loader that patches load_gguf_checkpoint stores the previous value
    as _orig_load_gguf_checkpoint in its own module globals.  We follow that
    chain until we reach a function that accepts 'model_to_load' as an
    explicit parameter — that is the genuine transformers implementation.
    """
    if _seen is None:
        _seen = set()
    if fn is None or id(fn) in _seen:
        return None
    _seen.add(id(fn))
    if not callable(fn):
        return None
    try:
        if "model_to_load" in _inspect.signature(fn).parameters:
            return fn
    except (TypeError, ValueError):
        pass
    # Follow the _orig_load_gguf_checkpoint chain through the function's module.
    # For gaiasky's own wrapper, also check _real_load_gguf_fn if present.
    globs = getattr(fn, "__globals__", {})
    for key in ("_orig_load_gguf_checkpoint", "_real_load_gguf_fn"):
        val = globs.get(key)
        if val is None:
            continue
        # _real_load_gguf_fn is a list holder; unwrap it.
        if isinstance(val, list):
            val = val[0] if val else None
        result = _find_real_load_gguf(val, _seen)
        if result is not None:
            return result
    return None


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 support and fix model_type."""
    _patch_qwen35_support()
    result = _real_load_gguf_fn[0](*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3"
    return result


def _apply_gguf_patches():
    """Re-apply patches immediately before calling transformers APIs.

    Other model loaders that run after us during pytest collection overwrite
    the module-level attribute with narrow-signature wrappers, and they may
    capture our wrapper as *their* _orig, chaining the problem further.
    We walk the __globals__ chain at call time to find the actual transformers
    implementation (the one with an explicit 'model_to_load' parameter) and
    store it in _real_load_gguf_fn so our wrapper always calls the right
    function regardless of import order.
    """
    _patch_qwen35_support()
    current = _gguf_utils.__dict__.get("load_gguf_checkpoint")
    if current is _patched_load_gguf_checkpoint:
        return  # Already installed; _real_load_gguf_fn is up-to-date.
    real = _find_real_load_gguf(current)
    if real is None:
        real = current  # current itself is the real function
    _real_load_gguf_fn[0] = real
    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


_apply_gguf_patches()

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
    """Available Gaiasky Qwen 3.5 GGUF model variants for causal language modeling."""

    GAIASKY_QWEN_3_5_9B_GGUF = "9B_GGUF"
    GAIASKY_QWEN_3_5_4B_GGUF = "4B_GGUF"


class ModelLoader(ForgeModel):
    """Gaiasky Qwen 3.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GAIASKY_QWEN_3_5_9B_GGUF: LLMModelConfig(
            pretrained_model_name="Langurmonkey/gaiasky-qwen-3.5-gguf",
            max_length=128,
        ),
        ModelVariant.GAIASKY_QWEN_3_5_4B_GGUF: LLMModelConfig(
            pretrained_model_name="Langurmonkey/gaiasky-qwen-3.5-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GAIASKY_QWEN_3_5_9B_GGUF

    _GGUF_FILES = {
        ModelVariant.GAIASKY_QWEN_3_5_9B_GGUF: "Qwen3.5-gaiasky-9B.Q4_K_M.gguf",
        ModelVariant.GAIASKY_QWEN_3_5_4B_GGUF: "Qwen3.5-gaiasky-4B.Q4_K_M.gguf",
    }

    sample_text = "Write a script to hide the stars and move the camera to Earth."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def gguf_file(self) -> str:
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Gaiasky Qwen 3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _apply_gguf_patches()
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
        _apply_gguf_patches()
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
        _apply_gguf_patches()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
