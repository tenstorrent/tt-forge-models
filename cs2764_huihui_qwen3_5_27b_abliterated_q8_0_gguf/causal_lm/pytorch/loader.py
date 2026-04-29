# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cs2764 Huihui Qwen3.5 27B Abliterated Q8_0 GGUF model loader implementation for causal language modeling.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    get_gguf_hf_weights_map as _orig_get_gguf_hf_weights_map,
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    GGUFTensor,
    TensorProcessor,
    TENSOR_PROCESSORS,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


# Config field mapping for the qwen35 GGUF architecture → Qwen3_5TextConfig fields.
# qwen35 is a hybrid SSM-Transformer (GatedDeltaNet + Attention) and is NOT the
# same as qwen3. The model class is Qwen3_5ForCausalLM (model_type qwen3_5_text).
_QWEN35_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.key_length": "head_dim",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    # SSM (GatedDeltaNet) layer configuration
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_key_head_dim",
    "ssm.group_count": "linear_num_key_heads",
    "ssm.time_step_rank": "linear_num_value_heads",
    # Layer type pattern: every full_attention_interval-th layer uses full attention
    "full_attention_interval": "full_attention_interval",
}


class _Qwen35TensorProcessor(TensorProcessor):
    """Tensor processor for qwen35 GGUF weights.

    - ssm_conv1d.weight: stored as [kernel_size, conv_dim]; HF expects [conv_dim, 1, kernel_size]
    - ssm_a: stored as A_log (log of positive A) — loaded directly, no transform needed
    """

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            # GGUF stores as [kernel_size, conv_dim] but the reader returns the numpy
            # array in C-order as [conv_dim, kernel_size].  HF Conv1d expects
            # [conv_dim, 1, kernel_size] (depthwise groups=conv_dim).
            weights = np.expand_dims(weights, axis=1)
        return GGUFTensor(weights, name, {})


def _patch_qwen35_support():
    """Register qwen35 as a fully supported GGUF architecture mapping to qwen3_5_text.

    qwen35 is a hybrid SSM-Transformer (GatedDeltaNet layers + full-attention layers).
    It maps to the transformers Qwen3_5ForCausalLM model (model_type qwen3_5_text),
    NOT to Qwen3ForCausalLM. The qwen3 alias used by older loaders is incorrect for
    this architecture because it ignores SSM layers and uses the wrong head_dim.
    """
    if "qwen35" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    # Register config mapping for qwen35 (do not overwrite if already the correct one)
    cfg_section = GGUF_TO_TRANSFORMERS_MAPPING["config"]
    if "qwen35" not in cfg_section or "ssm.conv_kernel" not in cfg_section.get("qwen35", {}):
        cfg_section["qwen35"] = _QWEN35_CONFIG_MAPPING

    # Copy tokenizer mappings from qwen3 (same tokenizer format)
    for section in ("tokenizer", "tokenizer_config"):
        sec = GGUF_TO_TRANSFORMERS_MAPPING[section]
        if "qwen3" in sec:
            sec.setdefault("qwen35", sec["qwen3"])

    # Register tokenizer fast converter
    if "qwen3" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen35", GGUF_TO_FAST_CONVERTERS["qwen3"])
        GGUF_TO_FAST_CONVERTERS.setdefault("qwen3_5_text", GGUF_TO_FAST_CONVERTERS["qwen3"])

    # Register tensor processor for SSM weight transforms
    TENSOR_PROCESSORS.setdefault("qwen35", _Qwen35TensorProcessor)


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen35 → qwen3_5_text support."""
    _patch_qwen35_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen35":
        result["config"]["model_type"] = "qwen3_5_text"
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
    """Wrap get_gguf_hf_weights_map to resolve qwen3_5_text → qwen35 for gguf arch lookup.

    Also injects missing dt_bias mappings: the gguf-py library maps ssm_dt → dt_proj
    but Qwen3_5GatedDeltaNet uses dt_bias (a plain Parameter) rather than dt_proj.bias.
    The injection is done only at the outermost call (qual_name == "") using the model's
    actual state_dict to avoid path-prefix confusion in recursive calls.
    """
    import re as _re
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    resolved_model_type = "qwen35" if model_type == "qwen3_5_text" else model_type
    result = _orig_get_gguf_hf_weights_map(hf_model, processor, resolved_model_type, num_layers, qual_name)

    # Inject at the outermost call only: iterate the model's actual state_dict to find all
    # dt_bias keys and create the GGUF blk.N.ssm_dt.bias → full HF path mapping.
    if resolved_model_type == "qwen35" and qual_name == "":
        _dt_pat = _re.compile(r"layers\.(\d+)\.linear_attn\.dt_bias$")
        for hf_name in hf_model.state_dict():
            m = _dt_pat.search(hf_name)
            if m:
                layer_idx = int(m.group(1))
                result[f"blk.{layer_idx}.ssm_dt.bias"] = hf_name

    return result


_patch_qwen35_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map

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
    """Available cs2764 Huihui Qwen3.5 27B Abliterated Q8_0 GGUF model variants for causal language modeling."""

    HUIHUI_QWEN3_5_27B_ABLITERATED_Q8_0_GGUF = "27B_Abliterated_Q8_0_GGUF"


class ModelLoader(ForgeModel):
    """cs2764 Huihui Qwen3.5 27B Abliterated Q8_0 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN3_5_27B_ABLITERATED_Q8_0_GGUF: LLMModelConfig(
            pretrained_model_name="cs2764/Huihui-Qwen3.5-27B-abliterated-Q8_0-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN3_5_27B_ABLITERATED_Q8_0_GGUF

    GGUF_FILE = "huihui-qwen3.5-27b-abliterated-q8_0.gguf"

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
            model="cs2764 Huihui Qwen3.5 27B Abliterated Q8_0 GGUF",
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

        messages = [{"role": "user", "content": self.sample_text}]
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
            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
