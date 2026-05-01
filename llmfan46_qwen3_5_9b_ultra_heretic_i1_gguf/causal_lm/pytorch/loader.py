# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
llmfan46-Qwen3.5-9B-ultra-heretic i1 GGUF model loader implementation for causal language modeling.
"""
import functools
import re as _re
import numpy as _np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    TensorProcessor as _TensorProcessor,
    GGUFTensor as _GGUFTensor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


class _Qwen35TensorProcessor(_TensorProcessor):
    """Fix qwen35-specific tensor name and shape mismatches during GGUF loading."""

    _DT_BIAS_RE = _re.compile(r"(?:model\.)?layers\.(\d+)\.linear_attn\.dt_bias$")

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map, suffix, qual_name, hf_name
    ):
        if m := self._DT_BIAS_RE.match(hf_name):
            n = m.group(1)
            gguf_to_hf_name_map[f"blk.{n}.ssm_dt.bias"] = qual_name + hf_name

    def process(self, weights, name, **kwargs):
        if name.endswith(".ssm_conv1d.weight") and weights.ndim == 2:
            weights = _np.expand_dims(weights, axis=1)
        return _GGUFTensor(weights, name, {})


def _patch_qwen35_support():
    """Register qwen35 GGUF architecture for loading as qwen3_5_text."""
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    config_mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get("config", {})
    if "qwen35" not in config_mapping:
        config_mapping["qwen35"] = {
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

    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"]
        )

    _gguf_utils.TENSOR_PROCESSORS.setdefault("qwen35", _Qwen35TensorProcessor)

    if getattr(_gguf_utils, "_qwen35_weights_map_patched", False):
        return
    _orig_weights_map_fn = _gguf_utils.get_gguf_hf_weights_map

    @functools.wraps(_orig_weights_map_fn)
    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None and hasattr(hf_model, "config"):
            model_type = hf_model.config.model_type
        if model_type == "qwen3_5_text":
            model_type = "qwen35"
        return _orig_weights_map_fn(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
    _gguf_utils._qwen35_weights_map_patched = True


def _get_raw_gguf_arch(gguf_path):
    """Read 'general.architecture' from GGUF metadata without going through any patch chain."""
    try:
        from array import array as _array
        from gguf import GGUFReader

        _reader = GGUFReader(gguf_path)
        _field = _reader.fields.get("general.architecture")
        if _field is None or not _field.data:
            return None
        return _array("B", list(_field.parts[_field.data[0]])).tobytes().decode()
    except Exception:
        return None


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 support and fix model_type."""
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
    if gguf_path and _get_raw_gguf_arch(gguf_path) == "qwen35":
        result["config"]["model_type"] = "qwen3_5_text"
    return result


_patch_qwen35_support()
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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
