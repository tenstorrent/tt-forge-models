# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gianni-cor bitnet_b1_58-large-TQ2_0 GGUF model loader implementation for causal language modeling.
"""

import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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


def _patch_transformers_bitnet_gguf():
    """Monkey-patch transformers to add bitnet GGUF architecture support.

    Transformers 5.x has BitNetForCausalLM but lacks GGUF loading support for
    the bitnet architecture. BitNet uses the same LLaMA-style tensor layout so
    we reuse LlamaTensorProcessor and GGUFLlamaConverter.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        LlamaTensorProcessor,
    )
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFLlamaConverter,
    )

    if "bitnet" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("bitnet")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["bitnet"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    }

    TENSOR_PROCESSORS["bitnet"] = LlamaTensorProcessor

    if "bitnet" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["bitnet"] = GGUFLlamaConverter


def _fix_gguf_load_compat():
    """Fix stale GGUF patches that omit model_to_load (added in transformers 5.x).

    Some other model loaders monkey-patch load_gguf_checkpoint with a signature
    that pre-dates the model_to_load parameter. We find the real transformers
    function (identified by an explicit model_to_load parameter — not just **kwargs)
    by recursively traversing closures and module globals, then replace the module
    attribute with a direct wrapper around it.
    """
    import transformers.modeling_gguf_pytorch_utils as gguf_utils_mod

    def _is_real(fn):
        """True only for the genuine transformers function with explicit model_to_load."""
        try:
            return "model_to_load" in inspect.signature(fn).parameters
        except (ValueError, TypeError):
            return False

    current = gguf_utils_mod.load_gguf_checkpoint
    if _is_real(current):
        return

    def _find_real(fn, visited=None, depth=0):
        if visited is None:
            visited = set()
        fn_id = id(fn)
        if fn_id in visited or depth > 60:
            return None
        visited.add(fn_id)

        if _is_real(fn):
            return fn

        # Check closures
        if hasattr(fn, "__closure__") and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    c = cell.cell_contents
                    if callable(c):
                        result = _find_real(c, visited, depth + 1)
                        if result is not None:
                            return result
                except ValueError:
                    pass

        # Check module globals — broken patches store _orig as a module-level global,
        # not as a closure variable. Search both "gguf" and "orig" in name to handle
        # varied naming conventions (_orig_load_gguf_checkpoint, orig_load, etc.).
        if hasattr(fn, "__globals__"):
            for name, val in fn.__globals__.items():
                if (
                    callable(val)
                    and ("gguf" in name.lower() or "orig" in name.lower())
                    and id(val) not in visited
                ):
                    result = _find_real(val, visited, depth + 1)
                    if result is not None:
                        return result

        return None

    real_fn = _find_real(current)
    if real_fn is None:
        return

    _real = real_fn

    def _compat(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        return _real(
            gguf_checkpoint_path,
            return_tensors=return_tensors,
            model_to_load=model_to_load,
        )

    gguf_utils_mod.load_gguf_checkpoint = _compat


_patch_transformers_bitnet_gguf()


class ModelVariant(StrEnum):
    """Available bitnet_b1_58-large-TQ2_0 GGUF model variants for causal language modeling."""

    BITNET_B1_58_LARGE_TQ2_0 = "bitnet_b1_58_large_TQ2_0"


class ModelLoader(ForgeModel):
    """bitnet_b1_58-large-TQ2_0 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BITNET_B1_58_LARGE_TQ2_0: LLMModelConfig(
            pretrained_model_name="gianni-cor/bitnet_b1_58-large-TQ2_0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BITNET_B1_58_LARGE_TQ2_0

    GGUF_FILE = "bitnet_b1_58-large-TQ2_0.gguf"

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
            model="bitnet_b1_58-large-TQ2_0 GGUF",
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
        _fix_gguf_load_compat()

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

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
