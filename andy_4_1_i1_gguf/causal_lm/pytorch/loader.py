# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Andy 4.1 i1 GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

# Andy-4.1 is a text-only fine-tune of Qwen3-VL whose GGUF uses the "qwen3vl"
# architecture identifier.  transformers does not yet register "qwen3vl" in its
# GGUF loader (only "qwen3" and "qwen3_moe" are present in 5.x).  Since the
# GGUF contains no visual encoder tensors the language-decoder weights are
# identical to a plain Qwen3 model.
#
# Two issues to fix:
# 1. "qwen3vl" missing from GGUF_CONFIG_MAPPING → ValueError on load.
# 2. The collection-wide GGUF patching chain (26+ loaders) results in an
#    outermost wrapper that does not accept the `model_to_load` kwarg added in
#    transformers 5.x, causing a TypeError on model loading.
#
# We fix (1) by registering the alias at import time, and fix (2) by installing
# a fresh outermost wrapper around each HF call that accepts `model_to_load`,
# drops it before forwarding (the inner chain ignores it anyway), and remaps
# model_type "qwen3vl" → "qwen3" in the returned config.
import contextlib
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.integrations import GGUF_CONFIG_MAPPING as _GGUF_CONFIG_MAPPING

if "qwen3vl" not in _GGUF_CONFIG_MAPPING and "qwen3" in _GGUF_CONFIG_MAPPING:
    _GGUF_CONFIG_MAPPING["qwen3vl"] = dict(_GGUF_CONFIG_MAPPING["qwen3"])
    if "qwen3vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")


def _find_real_load_gguf_checkpoint():
    """Walk the loader patching chain to find the original transformers function.

    Other loaders install wrappers around load_gguf_checkpoint at import time.
    Many of those wrappers do not accept the model_to_load kwarg added in
    transformers 5.x.  We bypass the whole chain and call the real function
    directly so that model_to_load (required when return_tensors=True) reaches
    the transformers implementation.
    """
    seen_ids = set()
    fn = _gguf_utils.load_gguf_checkpoint
    while id(fn) not in seen_ids:
        seen_ids.add(id(fn))
        if (getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils"
                and getattr(fn, "__qualname__", "") == "load_gguf_checkpoint"):
            return fn
        # Follow the captured inner function through the closure.
        found_next = None
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and "gguf_checkpoint" in getattr(val, "__name__", ""):
                found_next = val
                break
        if found_next is None:
            break
        fn = found_next
    return fn  # best effort — may be a wrapper if chain is unusual


_REAL_LOAD_GGUF_CHECKPOINT = _find_real_load_gguf_checkpoint()


@contextlib.contextmanager
def _qwen3vl_gguf_context():
    """Temporarily install a load_gguf_checkpoint wrapper that:
    - accepts model_to_load (added in transformers 5.x, missing from many
      loaders' wrappers in the global patching chain)
    - calls the original transformers function directly (bypassing the broken
      chain)
    - remaps model_type "qwen3vl" → "qwen3" in the returned config

    Several transformers modules import load_gguf_checkpoint at module level
    rather than looking it up dynamically, so we must patch each binding site.
    """
    import transformers.configuration_utils as _conf_utils
    import transformers.tokenization_utils_tokenizers as _tok_utils
    import transformers.models.auto.tokenization_auto as _tok_auto

    def _wrap(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        result = _REAL_LOAD_GGUF_CHECKPOINT(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )
        for section in ("config", "tokenizer_config"):
            if result.get(section, {}).get("model_type") == "qwen3vl":
                result[section]["model_type"] = "qwen3"
        return result

    _saved = {
        "_gguf_utils": _gguf_utils.load_gguf_checkpoint,
        "_conf_utils": _conf_utils.load_gguf_checkpoint,
        "_tok_utils": _tok_utils.load_gguf_checkpoint,
        "_tok_auto": _tok_auto.load_gguf_checkpoint,
    }
    _gguf_utils.load_gguf_checkpoint = _wrap
    _conf_utils.load_gguf_checkpoint = _wrap
    _tok_utils.load_gguf_checkpoint = _wrap
    _tok_auto.load_gguf_checkpoint = _wrap
    try:
        yield
    finally:
        _gguf_utils.load_gguf_checkpoint = _saved["_gguf_utils"]
        _conf_utils.load_gguf_checkpoint = _saved["_conf_utils"]
        _tok_utils.load_gguf_checkpoint = _saved["_tok_utils"]
        _tok_auto.load_gguf_checkpoint = _saved["_tok_auto"]

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
    """Available Andy 4.1 i1 GGUF model variants for causal language modeling."""

    ANDY_4_1_I1_GGUF = "4_1_I1_GGUF"


class ModelLoader(ForgeModel):
    """Andy 4.1 i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ANDY_4_1_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Andy-4.1-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANDY_4_1_I1_GGUF

    GGUF_FILE = "Andy-4.1.i1-Q4_K_M.gguf"

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
            model="Andy 4.1 i1 GGUF",
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

        with _qwen3vl_gguf_context():
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
            with _qwen3vl_gguf_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _qwen3vl_gguf_context():
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _qwen3vl_gguf_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
