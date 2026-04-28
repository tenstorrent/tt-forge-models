# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling.
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

# ---------------------------------------------------------------------------
# deepseek2 GGUF architecture patch
#
# The GigaChat3-10B-A1.8B GGUF file declares ``general.architecture =
# deepseek2``.  Transformers 5.x does not include deepseek2 in its GGUF
# config/tokenizer maps.
#
# Additionally, several other loaders in this repo install
# ``load_gguf_checkpoint`` wrappers with the fixed signature
# ``(gguf_path, return_tensors=False)`` that cannot accept the
# ``model_to_load`` keyword argument added in transformers 5.2.0.
# We detect and unwrap such broken wrappers at call time to ensure we
# always delegate to the real original.
# ---------------------------------------------------------------------------

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
)

# Register deepseek2 architecture (idempotent).
if "deepseek2" not in GGUF_SUPPORTED_ARCHITECTURES:
    GGUF_SUPPORTED_ARCHITECTURES.append("deepseek2")
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["deepseek2"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.freq_base": "rope_theta",
        "rope.dimension_count": "qk_rope_head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": None,
        "attention.value_length": None,
        "attention.key_length_mla": "qk_nope_head_dim",
        "attention.value_length_mla": "v_head_dim",
        "attention.q_lora_rank": "q_lora_rank",
        "attention.kv_lora_rank": "kv_lora_rank",
        "vocab_size": "vocab_size",
        "expert_count": "n_routed_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_shared_count": "n_shared_experts",
        "expert_group_count": "n_group",
        "expert_group_used_count": "topk_group",
        "expert_weights_scale": "routed_scaling_factor",
        "expert_weights_norm": "norm_topk_prob",
        "leading_dense_block_count": "first_k_dense_replace",
        "expert_feed_forward_length": "moe_intermediate_size",
    }

# Register tokenizer converters (idempotent via setdefault).
from transformers.integrations.ggml import (
    GGUF_TO_FAST_CONVERTERS,
    GGUFQwen2Converter,
)
GGUF_TO_FAST_CONVERTERS.setdefault("deepseek2", GGUFQwen2Converter)
# deepseek_v2 is needed because our wrapper rewrites model_type from
# "deepseek2" to "deepseek_v2" and the tokenizer path then calls
# convert_gguf_tokenizer("deepseek_v2", ...) which looks up this key.
GGUF_TO_FAST_CONVERTERS.setdefault("deepseek_v2", GGUFQwen2Converter)


def _unwrap_load_gguf_checkpoint(fn):
    """Return the real (unpatched) load_gguf_checkpoint by unwrapping broken wrappers.

    Several loaders in this repo install wrappers that cannot pass
    ``model_to_load`` through.  These wrappers capture the prior function
    either as a closure variable (``orig_load``, ``orig``) or as a module
    global (``_orig_load_gguf_checkpoint``).

    We walk the chain until we find a function that has ``model_to_load`` as
    an explicit named parameter (the real transformers original), skipping our
    own ``_deepseek2_load_gguf_checkpoint`` to avoid recursion.
    """
    visited = set()

    def _candidates(f):
        """Yield candidate 'next hop' callables from globals and closure."""
        # Module globals (some loaders store it here).
        fn_globals = getattr(f, "__globals__", {})
        for name in ("_orig_load_gguf_checkpoint", "orig_load", "orig"):
            v = fn_globals.get(name)
            if callable(v):
                yield v
        # Closure cells (most loaders use a local `orig_load = gguf_utils.load_gguf_checkpoint`).
        freevars = getattr(getattr(f, "__code__", None), "co_freevars", ())
        cells = getattr(f, "__closure__", None) or ()
        for varname, cell in zip(freevars, cells):
            if varname in ("orig_load", "orig", "_orig_load_gguf_checkpoint"):
                try:
                    v = cell.cell_contents
                    if callable(v):
                        yield v
                except ValueError:
                    pass

    while True:
        fn_id = id(fn)
        if fn_id in visited:
            break
        visited.add(fn_id)

        # Skip our own function to avoid returning ourselves as the "real" original.
        if fn is _deepseek2_load_gguf_checkpoint:
            inner = getattr(fn, "_underlying_load", None)
            if inner is None or not callable(inner) or inner is fn:
                break
            fn = inner
            continue

        # Check if this function accepts model_to_load as an explicit parameter.
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            if "model_to_load" in params:
                kind = params["model_to_load"].kind
                # Must be POSITIONAL_OR_KEYWORD or KEYWORD_ONLY (not VAR_KEYWORD).
                explicit_kinds = (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_ONLY,
                )
                if kind in explicit_kinds:
                    return fn  # Found the real original.
        except (ValueError, TypeError):
            break

        # Try to unwrap via captured original (globals or closure).
        next_fn = None
        for candidate in _candidates(fn):
            if id(candidate) not in visited and candidate is not fn:
                next_fn = candidate
                break
        if next_fn is None:
            break
        fn = next_fn

    return fn


def _deepseek2_load_gguf_checkpoint(*args, **kwargs):
    """Wrapper that normalises model_type deepseek2 -> deepseek_v2.

    Unwraps any broken load_gguf_checkpoint wrappers (those with fixed
    signature lacking model_to_load) to find the real original, then calls
    it with all arguments correctly passed through.
    """
    # Get the current global value - may be a broken wrapper.
    current_fn = _gguf_utils.load_gguf_checkpoint
    # If it's us (set by _install_deepseek2_gguf_patch), skip self to avoid
    # recursion and use the previously-stored _underlying_load.
    if current_fn is _deepseek2_load_gguf_checkpoint:
        real_fn = _deepseek2_load_gguf_checkpoint._underlying_load
    else:
        real_fn = _unwrap_load_gguf_checkpoint(current_fn)
    result = real_fn(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "deepseek2":
        config["model_type"] = "deepseek_v2"
    return result


def _deepseek2_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    """Wrapper that translates model_type deepseek_v2 -> deepseek2 for gguf-py arch lookup.

    The gguf-py library (MODEL_ARCH_NAMES) uses 'deepseek2' as the arch name,
    but transformers rewrites model_type to 'deepseek_v2'.  This wrapper
    translates back so get_gguf_hf_weights_map can find the arch.
    """
    if model_type is None:
        model_type = hf_model.config.model_type
    if model_type == "deepseek_v2":
        model_type = "deepseek2"
    orig = _deepseek2_get_gguf_hf_weights_map._underlying_map
    return orig(hf_model, processor, model_type, num_layers, qual_name)


def _install_deepseek2_gguf_patch():
    """(Re-)install the deepseek2 GGUF wrappers.

    Patches both load_gguf_checkpoint and get_gguf_hf_weights_map.

    Must be called immediately before any GGUF load to counteract other
    loaders that may have overwritten the globals with broken functions.
    Unwraps the current load_gguf_checkpoint to find the real original and
    stores it so that _deepseek2_load_gguf_checkpoint can call it without
    recursion.
    """
    import transformers.models.auto.tokenization_auto as _tok_auto
    import transformers.configuration_utils as _config_utils
    import transformers.tokenization_utils_tokenizers as _tok_utils

    # --- load_gguf_checkpoint patch ---
    # Find the real original before installing our wrapper.
    current = _gguf_utils.load_gguf_checkpoint
    if current is not _deepseek2_load_gguf_checkpoint:
        real = _unwrap_load_gguf_checkpoint(current)
    else:
        real = _deepseek2_load_gguf_checkpoint._underlying_load

    _deepseek2_load_gguf_checkpoint._underlying_load = real
    _gguf_utils.load_gguf_checkpoint = _deepseek2_load_gguf_checkpoint

    for _mod in (_tok_auto, _config_utils, _tok_utils):
        if hasattr(_mod, "load_gguf_checkpoint"):
            _mod.load_gguf_checkpoint = _deepseek2_load_gguf_checkpoint

    # --- get_gguf_hf_weights_map patch ---
    # Install idempotently: only wrap if not already our wrapper.
    current_map = _gguf_utils.get_gguf_hf_weights_map
    if current_map is not _deepseek2_get_gguf_hf_weights_map:
        _deepseek2_get_gguf_hf_weights_map._underlying_map = current_map
        _gguf_utils.get_gguf_hf_weights_map = _deepseek2_get_gguf_hf_weights_map


# Install at import time so the tokenizer lookup during collection works.
_install_deepseek2_gguf_patch()


class ModelVariant(StrEnum):
    """Available GigaChat3 10B A1.8B GGUF model variants for causal language modeling."""

    GIGACHAT3_10B_A1_8B_GGUF = "10B_A1_8B_GGUF"


class ModelLoader(ForgeModel):
    """GigaChat3 10B A1.8B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GIGACHAT3_10B_A1_8B_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/ai-sage_GigaChat3-10B-A1.8B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GIGACHAT3_10B_A1_8B_GGUF

    GGUF_FILE = "ai-sage_GigaChat3-10B-A1.8B-Q4_K_M.gguf"

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
            model="GigaChat3 10B A1.8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _install_deepseek2_gguf_patch()
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
        _install_deepseek2_gguf_patch()
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
