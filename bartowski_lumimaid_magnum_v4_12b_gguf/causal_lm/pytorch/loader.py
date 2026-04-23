# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski Lumimaid-Magnum-v4-12B GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils


def _find_real_load_gguf(fn):
    """Walk the monkey-patch chain to find the real transformers load_gguf_checkpoint.

    Patches are module-level functions that store the previous function in a
    module global (e.g. _orig_load_gguf_checkpoint). We walk that chain until
    we reach the function whose __module__ is transformers.modeling_gguf_pytorch_utils.
    """
    _ORIG_NAMES = (
        "_orig_load_gguf_checkpoint",
        "orig_load",
        "_orig",
        "_original_load_gguf",
    )
    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
            return fn
        # Most patches store the previous function in a module global.
        globs = getattr(fn, "__globals__", {})
        next_fn = None
        for key in _ORIG_NAMES:
            candidate = globs.get(key)
            if callable(candidate) and id(candidate) not in seen:
                next_fn = candidate
                break
        # Broader search: any global callable whose __name__ contains "load_gguf"
        if next_fn is None:
            for val in globs.values():
                if (
                    callable(val)
                    and "load_gguf" in getattr(val, "__name__", "")
                    and id(val) not in seen
                    and val is not fn
                ):
                    next_fn = val
                    break
        # Fall back to closure variables (nested-function style patches)
        if next_fn is None and fn.__closure__:
            freevars = getattr(fn.__code__, "co_freevars", ())
            for i, name in enumerate(freevars):
                if "load_gguf" in name:
                    try:
                        val = fn.__closure__[i].cell_contents
                        if callable(val) and id(val) not in seen:
                            next_fn = val
                            break
                    except ValueError:
                        pass
        if next_fn is None:
            return fn
        fn = next_fn
    return fn


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
    """Available bartowski Lumimaid-Magnum-v4-12B GGUF model variants for causal language modeling."""

    BARTOWSKI_LUMIMAID_MAGNUM_V4_12B_GGUF = "Lumimaid_Magnum_v4_12B_GGUF"


class ModelLoader(ForgeModel):
    """bartowski Lumimaid-Magnum-v4-12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BARTOWSKI_LUMIMAID_MAGNUM_V4_12B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Lumimaid-Magnum-v4-12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BARTOWSKI_LUMIMAID_MAGNUM_V4_12B_GGUF

    GGUF_FILE = "Lumimaid-Magnum-v4-12B-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

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
            model="bartowski Lumimaid-Magnum-v4-12B GGUF",
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

        # Other loaders monkey-patch load_gguf_checkpoint without accepting
        # model_to_load (added in transformers 5.x). Temporarily install a
        # wrapper that bypasses those patches and calls the real function.
        _outer = _gguf_utils.load_gguf_checkpoint
        _real = _find_real_load_gguf(_outer)

        def _compat_load_gguf(gguf_path, return_tensors=False, **kw):
            return _real(gguf_path, return_tensors=return_tensors, **kw)

        _gguf_utils.load_gguf_checkpoint = _compat_load_gguf
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _outer

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
