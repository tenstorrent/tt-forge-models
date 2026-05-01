# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.3 Nemotron Super 49B v1 GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
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


def _unwrap_to_true_load_gguf_checkpoint():
    """Traverse the patch chain to find transformers' original load_gguf_checkpoint.

    Other GGUF loaders (alphabetically earlier) patch _gguf_utils.load_gguf_checkpoint
    at import time in a chain. Many of those patches capture the previous-in-chain
    function as a module-level global (e.g. _orig_load_gguf_checkpoint). We walk that
    chain via __globals__ until we reach the function whose __module__ is
    'transformers.modeling_gguf_pytorch_utils' — the TRUE original function that
    accepts the model_to_load kwarg.
    """
    target = "transformers.modeling_gguf_pytorch_utils"
    visited: set = set()
    queue = [_gguf_utils.load_gguf_checkpoint]
    while queue:
        fn = queue.pop(0)
        fn_id = id(fn)
        if fn_id in visited:
            continue
        visited.add(fn_id)
        if getattr(fn, "__module__", "") == target:
            return fn
        # Follow closure cells (for nested-function patches)
        for cell in getattr(fn, "__closure__", None) or []:
            try:
                v = cell.cell_contents
                if callable(v):
                    queue.append(v)
            except ValueError:
                pass
        # Follow module-level globals that look like the captured original
        for k, v in list((getattr(fn, "__globals__", None) or {}).items()):
            if callable(v) and id(v) not in visited:
                k_lower = k.lower()
                if any(kw in k_lower for kw in ("orig", "true", "real", "load_gguf")):
                    queue.append(v)
    return _gguf_utils.load_gguf_checkpoint  # fallback


_true_load_gguf_checkpoint = _unwrap_to_true_load_gguf_checkpoint()


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint to remap deci model_type to llama.

    The Llama 3.3 Nemotron Super 49B GGUF file declares architecture as 'deci'
    (Deci AI's original name), but transformers has no 'deci' config class.
    The underlying architecture is llama-compatible, but uses per-layer arrays
    for attention head counts (hybrid attention/FFN-only layers). We collapse
    these to scalars by taking the max non-zero value.
    """
    kwargs = {"return_tensors": return_tensors}
    if model_to_load is not None:
        kwargs["model_to_load"] = model_to_load
    result = _true_load_gguf_checkpoint(gguf_path, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "deci":
        config["model_type"] = "llama"
        for field in (
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ):
            val = config.get(field)
            if isinstance(val, list):
                non_zero = [v for v in val if v]
                config[field] = max(non_zero) if non_zero else val[0]
    return result


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Llama 3.3 Nemotron Super 49B v1 GGUF model variants for causal language modeling."""

    NVIDIA_LLAMA_3_3_NEMOTRON_SUPER_49B_V1_Q4_K_M = (
        "Llama_3_3_Nemotron_Super_49B_v1_Q4_K_M"
    )


class ModelLoader(ForgeModel):
    """Llama 3.3 Nemotron Super 49B v1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NVIDIA_LLAMA_3_3_NEMOTRON_SUPER_49B_V1_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/nvidia_Llama-3_3-Nemotron-Super-49B-v1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NVIDIA_LLAMA_3_3_NEMOTRON_SUPER_49B_V1_Q4_K_M

    GGUF_FILE = "nvidia_Llama-3_3-Nemotron-Super-49B-v1-Q4_K_M.gguf"

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
            model="Llama 3.3 Nemotron Super 49B v1 GGUF",
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
        model_kwargs["ignore_mismatched_sizes"] = True

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # Re-apply patches: other loaders imported after us may have overwritten them.
        # modeling_utils.py does a local import of load_gguf_checkpoint, so it picks
        # up the current binding at call time.
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
