# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LucentLogico 3B i1 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata

import torch
from packaging import version as _version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    LlamaTensorProcessor,
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
)


class GraniteTensorProcessor(LlamaTensorProcessor):
    """LlamaTensorProcessor that normalizes per-layer list config values for Granite GGUF."""

    def process(self, weights, name, **kwargs):
        if ".attn_k." in name or ".attn_q." in name:
            for key in ("num_attention_heads", "num_key_value_heads"):
                val = self.config.get(key)
                if isinstance(val, list):
                    self.config[key] = max(val)
        return super().process(weights, name, **kwargs)


def _is_gguf_available_fixed(min_version: str = "0.10.0") -> bool:
    """Bypass stale PACKAGE_DISTRIBUTION_MAPPING by querying metadata directly."""
    try:
        gguf_version = importlib.metadata.version("gguf")
        return _version.parse(gguf_version) >= _version.parse(min_version)
    except Exception:
        return False


_gguf_utils.is_gguf_available = _is_gguf_available_fixed


def _patch_granite_support():
    """Register 'granite' as a supported GGUF architecture mapping to GraniteForCausalLM."""
    if "granite" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["granite"] = {
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
        "attention.scale": "attention_multiplier",
        "embedding_scale": "embedding_multiplier",
        "residual_scale": "residual_multiplier",
        "logit_scale": "logit_scale",
    }
    GGUF_SUPPORTED_ARCHITECTURES.append("granite")

    if "llama" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["granite"] = GGUF_TO_FAST_CONVERTERS["llama"]

    _gguf_utils.TENSOR_PROCESSORS["granite"] = GraniteTensorProcessor


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Patch load_gguf_checkpoint to support granite architecture.

    Applies granite architecture support and flattens the per-layer
    num_key_value_heads list that Granite GGUF files store.
    """
    _patch_granite_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "granite":
        kv_heads = result["config"].get("num_key_value_heads")
        if isinstance(kv_heads, list):
            result["config"]["num_key_value_heads"] = max(kv_heads)
    return result


_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


def _find_original_gguf_checkpoint_fn():
    """Find the original load_gguf_checkpoint from transformers.

    Module-level patch functions store their chain reference as a module global
    (LOAD_GLOBAL, not a closure cell), so traverse __globals__ as fallback.
    """
    visited = set()
    fn = _gguf_utils.load_gguf_checkpoint
    while fn is not None:
        fn_id = id(fn)
        if fn_id in visited:
            return None
        visited.add(fn_id)
        if getattr(fn, "__module__", "") == "transformers.modeling_gguf_pytorch_utils":
            return fn
        next_fn = None
        if hasattr(fn, "__closure__") and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                    if callable(val) and not isinstance(val, type):
                        next_fn = val
                        break
                except ValueError:
                    pass
        if next_fn is None:
            next_fn = getattr(fn, "__globals__", {}).get("_orig_load_gguf_checkpoint")
        fn = next_fn
    return None


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


class ModelVariant(StrEnum):
    """Available LucentLogico 3B i1 GGUF model variants for causal language modeling."""

    LUCENTLOGICO_3B_I1_GGUF = "LucentLogico_3B_i1_GGUF"


class ModelLoader(ForgeModel):
    """LucentLogico 3B i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LUCENTLOGICO_3B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/LucentLogico-3B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LUCENTLOGICO_3B_I1_GGUF

    GGUF_FILE = "LucentLogico-3B.i1-Q4_K_M.gguf"

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
            model="LucentLogico 3B i1 GGUF",
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

        # The outer patch chain (e.g. mradermacher loaders) uses fixed signatures
        # that don't forward model_to_load to the original load_gguf_checkpoint.
        # Install a runtime wrapper as the outermost so model_to_load reaches the
        # real transformers function for tensor loading.
        _saved_chain = _gguf_utils.load_gguf_checkpoint
        _real_orig = _find_original_gguf_checkpoint_fn()

        if _real_orig is not None:

            def _model_to_load_compat(*args, **kw):
                model_to_load = kw.pop("model_to_load", None)
                if model_to_load is not None:
                    return _real_orig(*args, model_to_load=model_to_load, **kw)
                return _saved_chain(*args, **kw)

            _gguf_utils.load_gguf_checkpoint = _model_to_load_compat

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _saved_chain

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
