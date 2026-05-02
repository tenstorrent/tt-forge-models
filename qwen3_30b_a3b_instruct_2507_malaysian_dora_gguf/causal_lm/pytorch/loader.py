# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-30B-A3B-Instruct-2507 Malaysian DoRA GGUF model loader implementation for causal language modeling.
"""
import importlib
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils

_GGUF_UTILS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(transformers.__file__), "modeling_gguf_pytorch_utils.py")
)


def _find_real_load_gguf_checkpoint(func):
    """Follow the narrow-sig wrapper chain to the real transformers load_gguf_checkpoint.

    Other loaders patch _gguf_utils.load_gguf_checkpoint with narrow-sig wrappers that
    call _orig_load_gguf_checkpoint from their module globals (LOAD_GLOBAL, not closure).
    We trace through those global references until we reach the function whose __code__
    lives in the transformers source file.
    """
    seen = set()
    while func is not None:
        fid = id(func)
        if fid in seen:
            return None
        seen.add(fid)
        code = getattr(func, "__code__", None)
        if code and os.path.abspath(code.co_filename) == _GGUF_UTILS_FILE:
            return func
        # Narrow-sig loaders store the previous function as _orig_load_gguf_checkpoint
        # in their module's global namespace (accessed via LOAD_GLOBAL, not closure).
        globals_ = getattr(func, "__globals__", {})
        next_func = globals_.get("_orig_load_gguf_checkpoint")
        if next_func is None or not callable(next_func):
            return None
        func = next_func
    return None


# Capture whatever was in the module before we install our patch (may be a
# narrow-sig wrapper from another loader imported earlier during test collection).
_before_our_patch = _gguf_utils.__dict__.get("load_gguf_checkpoint")
_real_load_gguf = _find_real_load_gguf_checkpoint(_before_our_patch)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Pass through to real transformers load_gguf_checkpoint, accepting model_to_load kwarg."""
    if _real_load_gguf is not None:
        return _real_load_gguf(*args, **kwargs)
    # Fallback: reload the module to restore the unpatched original.
    importlib.reload(_gguf_utils)
    return _gguf_utils.load_gguf_checkpoint(*args, **kwargs)


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
    """Available Qwen3-30B-A3B-Instruct-2507 Malaysian DoRA GGUF model variants for causal language modeling."""

    QWEN_3_30B_A3B_INSTRUCT_2507_MALAYSIAN_DORA_GGUF = (
        "30B_A3B_Instruct_2507_Malaysian_DoRA_GGUF"
    )


class ModelLoader(ForgeModel):
    """Qwen3-30B-A3B-Instruct-2507 Malaysian DoRA GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_30B_A3B_INSTRUCT_2507_MALAYSIAN_DORA_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-30B-A3B-Instruct-2507-Malaysian-DoRA-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_30B_A3B_INSTRUCT_2507_MALAYSIAN_DORA_GGUF

    GGUF_FILE = "Qwen3-30B-A3B-Instruct-2507-Malaysian-DoRA.Q4_K_M.gguf"

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
            model="Qwen3-30B-A3B-Instruct-2507 Malaysian DoRA GGUF",
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

        # Re-apply patch here (not just at import time) so other narrow-sig loaders
        # imported after us cannot clobber it before load_model is called.
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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

        # Qwen3MoeExperts.forward uses a Python for-loop over a dynamically-sized
        # expert_hit tensor that XLA/torch.compile cannot statically trace, causing
        # a segfault in partition_fx_graph_for_cpu_fallback. batched_mm uses only
        # static tensor operations and is fully XLA-compatible.
        if hasattr(model, "config"):
            model.config._experts_implementation = "batched_mm"

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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = (
                    "model",
                    "batch",
                )
                shard_specs[mlp.shared_expert.down_proj.weight] = (
                    "batch",
                    "model",
                )

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
