# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boto 9B i1 GGUF model loader implementation for causal language modeling.
"""

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

_ORIG_VAR_NAMES = (
    "_orig_load_gguf_checkpoint",
    "orig_load",
    "_orig_load",
    "_original",
    "original",
)


def _find_true_original(fn):
    """Walk the patch chain to find the real transformers load_gguf_checkpoint."""
    seen = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
            return fn
        globs = getattr(fn, "__globals__", {}) or {}
        next_fn = None
        for name in _ORIG_VAR_NAMES:
            candidate = globs.get(name)
            if (
                candidate is not None
                and callable(candidate)
                and id(candidate) not in seen
            ):
                next_fn = candidate
                break
        if next_fn is not None:
            fn = next_fn
            continue
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if callable(val) and id(val) not in seen:
                next_fn = val
                break
        if next_fn is not None:
            fn = next_fn
            continue
        break
    return fn


def _restore_original_gguf_checkpoint():
    """Restore the true original load_gguf_checkpoint, bypassing broken patches."""
    true_orig = _find_true_original(_gguf_utils.load_gguf_checkpoint)
    _gguf_utils.load_gguf_checkpoint = true_orig
    _config_utils.load_gguf_checkpoint = true_orig
    _auto_tokenizer.load_gguf_checkpoint = true_orig
    _tok_utils.load_gguf_checkpoint = true_orig


class ModelVariant(StrEnum):
    """Available Boto 9B i1 GGUF model variants for causal language modeling."""

    BOTO_9B_I1_GGUF = "9B_I1_GGUF"


class ModelLoader(ForgeModel):
    """Boto 9B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BOTO_9B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/boto-9B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BOTO_9B_I1_GGUF

    GGUF_FILE = "boto-9B.i1-Q4_K_M.gguf"

    sample_text = "Qual é a sua cidade favorita?"

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
            model="Boto 9B i1 GGUF",
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
        _restore_original_gguf_checkpoint()
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

        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
        _restore_original_gguf_checkpoint()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
