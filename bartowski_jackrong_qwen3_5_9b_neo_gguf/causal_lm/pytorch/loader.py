# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski Jackrong Qwen3.5-9B-Neo GGUF model loader implementation for causal language modeling.

Qwen3.5-9B-Neo is a hybrid SSM/full-attention model (GGUF architecture "qwen35")
that alternates GatedDeltaNet linear-attention layers with periodic full-attention
layers every 4 blocks.  Transformers 5.2 ships Qwen3_5ForCausalLM which models
this architecture, but the GGUF loader has no config-field or tensor-name mapping
for "qwen35".  This loader monkey-patches that gap.
"""
import importlib.metadata
import re
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
    load_gguf_checkpoint as _captured_load_gguf_checkpoint,
)


def _find_real_load_gguf():
    """Walk the patch chain to find the real transformers load_gguf_checkpoint.

    Other GGUF loaders imported earlier (e.g. bartowski_coniccat) replace
    the module attribute with a fixed-signature wrapper, then capture the
    previous function in their own _orig_load_gguf_checkpoint global.  Walking
    the __globals__ chain reaches the real transformers function that accepts
    model_to_load and other kwargs added in transformers 5.x.
    """
    fn = _captured_load_gguf_checkpoint
    seen: set = set()
    while True:
        fn_id = id(fn)
        if fn_id in seen:
            break
        seen.add(fn_id)
        next_fn = (
            fn.__globals__.get("_orig_load_gguf_checkpoint")
            if hasattr(fn, "__globals__")
            else None
        )
        if next_fn is None or id(next_fn) in seen:
            break
        fn = next_fn
    return fn


_orig_load_gguf_checkpoint = _find_real_load_gguf()

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
    "rope.dimension_count": None,       # covered by partial_rotary_factor default
    "rope.dimension_sections": None,    # VLM-only; unused for text GGUF
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "attention.key_length": "head_dim",
    "attention.value_length": None,     # same as head_dim; redundant
    # SSM / GatedDeltaNet parameters
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_value_head_dim",      # head_v_dim
    "ssm.group_count": "linear_num_key_heads",      # num_k_heads
    "ssm.time_step_rank": "linear_num_value_heads", # num_v_heads
    "ssm.inner_size": None,             # redundant (= num_v_heads * head_v_dim)
    # Hybrid schedule: Qwen3_5TextConfig converts this to a layer_types list
    "full_attention_interval": "full_attention_interval",
    "vocab_size": "vocab_size",
}


# ── Tensor processor ─────────────────────────────────────────────────────────

class _Qwen35TensorProcessor(TensorProcessor):
    """Map GGUF "qwen35" tensor names to Qwen3_5ForCausalLM HF state-dict names.

    gguf-py's get_tensor_name_map returns a pass-through (no-op) for arch 34
    ("qwen35"), so every tensor falls through to perform_fallback_tensor_mapping.
    """

    _LAYER_PATTERNS: list[tuple] = [
        # Linear-attention (GatedDeltaNet) layers
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.in_proj_qkv$"), "attn_qkv", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.in_proj_z$"),   "attn_gate", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.in_proj_b$"),   "ssm_beta", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.in_proj_a$"),   "ssm_alpha", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.conv1d$"),      "ssm_conv1d", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.dt_bias$"),     "ssm_dt.bias", False),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.A_log$"),       "ssm_a", False),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.norm$"),        "ssm_norm", True),
        (re.compile(r"model\.layers\.(\d+)\.linear_attn\.out_proj$"),    "ssm_out", True),
        # Full-attention layers
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.q_proj$"),        "attn_q", True),
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.k_proj$"),        "attn_k", True),
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.v_proj$"),        "attn_v", True),
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.o_proj$"),        "attn_output", True),
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.q_norm$"),        "attn_q_norm", True),
        (re.compile(r"model\.layers\.(\d+)\.self_attn\.k_norm$"),        "attn_k_norm", True),
        # Shared across all layer types
        (re.compile(r"model\.layers\.(\d+)\.input_layernorm$"),          "attn_norm", True),
        (re.compile(r"model\.layers\.(\d+)\.post_attention_layernorm$"), "post_attention_norm", True),
        (re.compile(r"model\.layers\.(\d+)\.mlp\.gate_proj$"),           "ffn_gate", True),
        (re.compile(r"model\.layers\.(\d+)\.mlp\.up_proj$"),             "ffn_up", True),
        (re.compile(r"model\.layers\.(\d+)\.mlp\.down_proj$"),           "ffn_down", True),
    ]

    _GLOBAL_MAP: dict[str, tuple] = {
        "model.embed_tokens": ("token_embd", True),
        "model.norm":         ("output_norm", True),
        "lm_head":            ("output", True),
    }

    def perform_fallback_tensor_mapping(
        self,
        gguf_to_hf_name_map: dict,
        suffix: str,
        qual_name: str,
        hf_name: str,
    ) -> None:
        base_name = hf_name[: -len(suffix)] if suffix else hf_name

        for pattern, gguf_comp, append_suffix in self._LAYER_PATTERNS:
            m = pattern.match(base_name)
            if m:
                bid = m.group(1)
                gguf_key = f"blk.{bid}.{gguf_comp}"
                if append_suffix:
                    gguf_key += suffix
                gguf_to_hf_name_map[gguf_key] = qual_name + hf_name
                return

        for hf_prefix, (gguf_comp, append_suffix) in self._GLOBAL_MAP.items():
            if base_name == hf_prefix:
                gguf_to_hf_name_map[gguf_comp + (suffix if append_suffix else "")] = (
                    qual_name + hf_name
                )
                return

    def process(self, weights: np.ndarray, name: str, **kwargs) -> GGUFTensor:
        if "ssm_a" in name and ".weight" not in name:
            # GGUF stores ssm_a as raw positive values following -exp(A_log)
            # convention; convert to A_log = log(-ssm_a).
            weights = np.log(-weights)
        elif "ssm_conv1d.weight" in name:
            # After dequantize, shape is (out_channels, kernel_size).
            # PyTorch depthwise Conv1d expects (out_channels, 1, kernel_size).
            weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


# ── Patch installation ────────────────────────────────────────────────────────

_orig_get_gguf_hf_weights_map = None


def _patch_qwen35_support() -> None:
    """Register qwen35 GGUF architecture with the transformers GGUF loader."""
    global _orig_get_gguf_hf_weights_map
    from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES

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
            if model_type is None:
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


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 → qwen3_5_text support."""
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if "full_attention_interval" in result.get("config", {}):
        result["config"]["model_type"] = "qwen3_5_text"
    return result


_patch_qwen35_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


# ── Model definition ──────────────────────────────────────────────────────────


class ModelVariant(StrEnum):
    """Available bartowski Jackrong Qwen3.5-9B-Neo GGUF model variants for causal language modeling."""

    JACKRONG_QWEN3_5_9B_NEO_Q4_K_M = "Jackrong_Qwen3_5_9B_Neo_Q4_K_M"


class ModelLoader(ForgeModel):
    """bartowski Jackrong Qwen3.5-9B-Neo GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.JACKRONG_QWEN3_5_9B_NEO_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/Jackrong_Qwen3.5-9B-Neo-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JACKRONG_QWEN3_5_9B_NEO_Q4_K_M

    GGUF_FILE = "Jackrong_Qwen3.5-9B-Neo-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    @staticmethod
    def _fix_gguf_version_detection():
        """Fix stale PACKAGE_DISTRIBUTION_MAPPING when gguf is installed late.

        transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time. When gguf
        is installed later, the mapping is stale and version detection falls back to
        gguf.__version__ which doesn't exist, yielding 'N/A' and crashing version.parse.
        """
        import transformers.utils.import_utils as _import_utils

        if "gguf" not in _import_utils.PACKAGE_DISTRIBUTION_MAPPING:
            try:
                importlib.metadata.version("gguf")
                _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = ["gguf"]
                _import_utils.is_gguf_available.cache_clear()
            except importlib.metadata.PackageNotFoundError:
                pass

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
            model="bartowski Jackrong Qwen3.5-9B-Neo GGUF",
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
        self._fix_gguf_version_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
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
        self._fix_gguf_version_detection()
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts = [text]
        else:
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
            if hasattr(layer, "mlp"):
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self._fix_gguf_version_detection()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
