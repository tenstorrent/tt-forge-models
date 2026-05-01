# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 350m GGUF model loader implementation for causal language modeling.
"""
import contextlib
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


def _find_real_load_gguf():
    """Walk the load_gguf_checkpoint patch chain to find the real transformers function.

    Other loaders may install wrappers that lack **kwargs and silently drop model_to_load.
    We walk the chain by inspecting each wrapper's globals AND closure cells, stopping when
    we reach the genuine transformers implementation (identified by its __module__).
    Different loaders use different variable names and different scoping (module-level vs
    nested-function closure), so we probe both.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils

    _CAPTURED_NAMES = frozenset((
        "_orig_load_gguf_checkpoint",
        "orig_load",
        "_orig",
        "_real_load_gguf_checkpoint",
    ))

    def _find_next(fn):
        # Check module-level globals first (e.g. `from ... import X as _orig_load_gguf_checkpoint`)
        globs = getattr(fn, "__globals__", {})
        for name in _CAPTURED_NAMES:
            candidate = globs.get(name)
            if candidate is not None and callable(candidate) and candidate is not fn:
                return candidate
        # Check closure cells (e.g. `orig_load = gguf_utils.load_gguf_checkpoint` inside a
        # helper function, captured by a nested wrapper)
        closure = getattr(fn, "__closure__", None) or ()
        freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ())
        for cell, name in zip(closure, freevars):
            if name not in _CAPTURED_NAMES:
                continue
            try:
                candidate = cell.cell_contents
            except ValueError:
                continue
            if callable(candidate) and candidate is not fn:
                return candidate
        return None

    fn = _gguf_utils.load_gguf_checkpoint
    visited = set()
    while getattr(fn, "__module__", None) != "transformers.modeling_gguf_pytorch_utils":
        fid = id(fn)
        if fid in visited:
            break
        visited.add(fid)
        next_fn = _find_next(fn)
        if next_fn is None:
            break
        fn = next_fn
    return fn


def _patch_transformers_granite_gguf():
    """Register granite GGUF architecture in transformers.

    Transformers 5.x has GraniteForCausalLM but lacks GGUF loading support for the
    'granite' architecture. This patch registers the config field mapping, the tokenizer
    converter, and the supported-architecture entry.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES

    if "granite" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("granite")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["granite"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "attention.scale": "attention_multiplier",
        "embedding_scale": "embedding_multiplier",
        "residual_scale": "residual_multiplier",
        "logit_scale": "logits_scaling",
    }

    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter

    if "granite" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granite"] = GGUFGPTConverter


_patch_transformers_granite_gguf()


@contextlib.contextmanager
def _granite_gguf_load_context():
    """Context manager that installs a correct load_gguf_checkpoint for granite.

    Later-imported loaders with bad (gguf_path, return_tensors=False) signatures
    overwrite the module attribute and silently drop model_to_load. This context
    manager temporarily installs a wrapper that:
      - Calls the real transformers load_gguf_checkpoint directly (bypassing broken
        intermediate wrappers) so that model_to_load is correctly forwarded.
      - Post-processes the granite config to convert the per-layer num_key_value_heads
        array (28 identical values) to the scalar GraniteConfig expects.
    Saves and restores all module bindings so other tests are unaffected.
    """
    import transformers.modeling_gguf_pytorch_utils as _gguf_utils
    import transformers.configuration_utils as _config_utils
    import transformers.models.auto.tokenization_auto as _auto_tokenizer
    import transformers.tokenization_utils_tokenizers as _tok_utils

    real_fn = _find_real_load_gguf()

    def _patched(gguf_path, return_tensors=False, **kwargs):
        result = real_fn(gguf_path, return_tensors=return_tensors, **kwargs)
        if result.get("config", {}).get("model_type") == "granite":
            kv = result["config"].get("num_key_value_heads")
            if isinstance(kv, list):
                result["config"]["num_key_value_heads"] = max(kv)
        return result

    mods = [_gguf_utils, _config_utils, _auto_tokenizer, _tok_utils]
    saved = {mod: getattr(mod, "load_gguf_checkpoint", None) for mod in mods}
    for mod in mods:
        mod.load_gguf_checkpoint = _patched
    try:
        yield
    finally:
        for mod, orig in saved.items():
            if orig is not None:
                mod.load_gguf_checkpoint = orig


class ModelVariant(StrEnum):
    """Available Granite 4.0 350m GGUF model variants for causal language modeling."""

    GRANITE_4_0_350M_Q4_K_M_GGUF = "granite_4_0_350m_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Granite 4.0 350m GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_350M_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-350m-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_350M_Q4_K_M_GGUF

    GGUF_FILE = "granite-4.0-350m-Q4_K_M.gguf"

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Granite 4.0 350m GGUF",
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
            with _granite_gguf_load_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _granite_gguf_load_context():
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
        with _granite_gguf_load_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
