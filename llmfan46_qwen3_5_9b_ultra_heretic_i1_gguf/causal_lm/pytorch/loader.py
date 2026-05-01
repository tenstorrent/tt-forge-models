# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
llmfan46-Qwen3.5-9B-ultra-heretic i1 GGUF model loader implementation for causal language modeling.

Qwen3.5-9B is a hybrid SSM/full-attention model (GGUF architecture "qwen35").
The GGUF loader has no config-field or tensor-name mapping for "qwen35".  This
loader monkey-patches that gap via a context manager so other GGUF loaders that
patch the same binding sites do not interfere at run time.
"""
import re
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUFTensor,
    TensorProcessor,
    TENSOR_PROCESSORS,
    GGUF_SUPPORTED_ARCHITECTURES,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# ── GGUF config-field mapping for the "qwen35" architecture ─────────────────

_QWEN35_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "attention.key_length": "head_dim",
    "vocab_size": "vocab_size",
    "full_attention_interval": "full_attention_interval",
}


# ── Tensor processor ─────────────────────────────────────────────────────────

class _Qwen35TensorProcessor(TensorProcessor):
    """Fix qwen35-specific tensor name and shape mismatches during GGUF loading."""

    _DT_BIAS_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.linear_attn\.dt_bias$")

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        if m := self._DT_BIAS_RE.match(hf_name):
            n = m.group(1)
            gguf_to_hf_name_map[f"blk.{n}.ssm_dt.bias"] = qual_name + hf_name

    def process(self, weights, name, **kwargs):
        if name.endswith(".ssm_conv1d.weight") and weights.ndim == 2:
            weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


# ── Registry patches (idempotent, applied once at import time) ────────────────

_orig_get_gguf_hf_weights_map = None


def _patch_qwen35_registry() -> None:
    """Register qwen35 arch in GGUF tables (idempotent, safe to call repeatedly)."""
    global _orig_get_gguf_hf_weights_map

    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    cfg_map = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]
    cfg_map["qwen35"] = _QWEN35_CONFIG_MAPPING

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )

    TENSOR_PROCESSORS["qwen35"] = _Qwen35TensorProcessor

    if _orig_get_gguf_hf_weights_map is None:
        _orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

        def _patched_get_gguf_hf_weights_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if model_type is None and hasattr(hf_model, "config"):
                model_type = hf_model.config.model_type
            if model_type == "qwen3_5_text":
                model_type = "qwen35"
            return _orig_get_gguf_hf_weights_map(
                hf_model,
                processor,
                model_type=model_type,
                num_layers=num_layers,
                qual_name=qual_name,
            )

        _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen35_registry()


# ── Context manager for the load_gguf_checkpoint patch ───────────────────────

def _find_real_load_gguf_at_call_time():
    """BFS over the load_gguf_checkpoint patch chain to find the real function.

    Multiple loaders patch _gguf_utils.load_gguf_checkpoint at import time; the
    last importer wins alphabetically.  Each wrapper captures its predecessor
    as either a module global or a closure cell.  We BFS over both until we find
    a function whose __globals__ IS the _gguf_utils module dict — that function
    is defined in transformers.modeling_gguf_pytorch_utils and is the real impl.
    """
    _GLOBAL_VARS = (
        "_orig_load_gguf_checkpoint",
        "orig_load",
        "_orig_load",
    )
    seen: set = set()
    queue: list = [_gguf_utils.load_gguf_checkpoint]

    while queue:
        fn = queue.pop(0)
        if not callable(fn):
            continue
        fn_id = id(fn)
        if fn_id in seen:
            continue
        seen.add(fn_id)

        if hasattr(fn, "__globals__") and fn.__globals__ is vars(_gguf_utils):
            return fn

        if hasattr(fn, "__globals__"):
            for var in _GLOBAL_VARS:
                candidate = fn.__globals__.get(var)
                if candidate is not None and callable(candidate) and id(candidate) not in seen:
                    queue.append(candidate)

        if getattr(fn, "__closure__", None):
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and id(val) not in seen:
                        queue.append(val)
                except ValueError:
                    pass

    return _gguf_utils.load_gguf_checkpoint  # fallback


@contextmanager
def _qwen35_gguf_context():
    """Temporarily install the qwen35-aware load_gguf_checkpoint wrapper.

    Applied at call time (not import time) so our patch is in effect regardless
    of which other loaders were imported after us.
    """
    _patch_qwen35_registry()
    real_fn = _find_real_load_gguf_at_call_time()

    def _my_load_gguf_checkpoint(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        if "full_attention_interval" in result.get("config", {}):
            result["config"]["model_type"] = "qwen3_5_text"
        return result

    _binding_sites = [
        (_gguf_utils, "load_gguf_checkpoint"),
        (_config_utils, "load_gguf_checkpoint"),
        (_auto_tokenizer, "load_gguf_checkpoint"),
        (_tok_utils, "load_gguf_checkpoint"),
    ]
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr in _binding_sites]
    for mod, attr in _binding_sites:
        setattr(mod, attr, _my_load_gguf_checkpoint)
    try:
        yield
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)


class ModelVariant(StrEnum):
    """Available llmfan46-Qwen3.5-9B-ultra-heretic i1 GGUF model variants for causal language modeling."""

    LLMFAN46_QWEN3_5_9B_ULTRA_HERETIC_I1_GGUF = "9B_ultra_heretic_i1_GGUF"


class ModelLoader(ForgeModel):
    """llmfan46-Qwen3.5-9B-ultra-heretic i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLMFAN46_QWEN3_5_9B_ULTRA_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/llmfan46-Qwen3.5-9B-ultra-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLMFAN46_QWEN3_5_9B_ULTRA_HERETIC_I1_GGUF

    GGUF_FILE = "llmfan46-Qwen3.5-9B-ultra-heretic.i1-Q4_K_M.gguf"

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
            model="llmfan46-Qwen3.5-9B-ultra-heretic i1 GGUF",
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

        with _qwen35_gguf_context():
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
            with _qwen35_gguf_context():
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.GGUF_FILE
                )
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with _qwen35_gguf_context():
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

        # Qwen3_5DynamicCache is not a pytree-registered type; use_cache=False
        # prevents it from appearing in the output without affecting logits.
        inputs["use_cache"] = False

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

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        with _qwen35_gguf_context():
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
            )
        return self.config
