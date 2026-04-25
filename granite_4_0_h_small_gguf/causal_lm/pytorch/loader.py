# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite 4.0 H-Small GGUF model loader implementation for causal language modeling.
"""

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
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

_GRANITEHYBRID_CONFIG_MAP = {
    "block_count": "num_hidden_layers",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "rope.freq_base": "rope_theta",
    "rope.dimension_count": "head_dim",
    "vocab_size": "vocab_size",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_shared_feed_forward_length": "shared_intermediate_size",
    "attention.scale": "attention_multiplier",
    "embedding_scale": "embedding_multiplier",
    "residual_scale": "residual_multiplier",
    "logit_scale": "logits_scaling",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.state_size": "mamba_d_state",
    "ssm.group_count": "mamba_n_groups",
    "ssm.time_step_rank": "mamba_n_heads",
}


def _patch_granitehybrid_support():
    """Register granitehybrid GGUF architecture as granitemoehybrid in transformers."""
    if "granitehybrid" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "granitehybrid", _GRANITEHYBRID_CONFIG_MAP
    )
    # granitehybrid uses a GPT-2 style BPE tokenizer
    if "gpt2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "granitemoehybrid", GGUF_TO_FAST_CONVERTERS["gpt2"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "granitehybrid", GGUF_TO_FAST_CONVERTERS["gpt2"]
        )


def _infer_granitehybrid_layer_types(gguf_path, num_layers):
    """Determine per-layer type (attention vs mamba) from tensor names in the GGUF file."""
    import gguf as _gguf

    reader = _gguf.GGUFReader(gguf_path)
    mamba_layers = set()
    for tensor in reader.tensors:
        if tensor.name.startswith("blk."):
            parts = tensor.name.split(".")
            block_num = int(parts[1])
            if len(parts) > 2 and parts[2] == "ssm_a":
                mamba_layers.add(block_num)
    return ["mamba" if i in mamba_layers else "attention" for i in range(num_layers)]


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    """Wrap load_gguf_checkpoint to add granitehybrid GGUF support."""
    _patch_granitehybrid_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
    cfg = result.get("config", {})
    if cfg.get("model_type") == "granitehybrid":
        cfg["model_type"] = "granitemoehybrid"
        cfg["architectures"] = ["GraniteMoeHybridForCausalLM"]
        # GGUF stores per-layer KV head counts (0 for mamba, N for attention layers).
        # Extract the max non-zero value as the single num_key_value_heads for the config.
        kv_heads = cfg.get("num_key_value_heads")
        if isinstance(kv_heads, list):
            cfg["num_key_value_heads"] = max(
                (v for v in kv_heads if v > 0),
                default=cfg.get("num_attention_heads", 32),
            )
        elif not kv_heads:
            cfg["num_key_value_heads"] = cfg.get("num_attention_heads", 32)
        # Derive per-layer block types (attention vs mamba) from tensor names
        num_layers = cfg.get("num_hidden_layers", 40)
        cfg["layer_types"] = _infer_granitehybrid_layer_types(gguf_path, num_layers)
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kwargs):
    """Wrap get_gguf_hf_weights_map to map granitemoehybrid back to granitehybrid for gguf-py."""
    resolved = (
        model_type
        if model_type is not None
        else getattr(hf_model.config, "model_type", None)
    )
    if resolved == "granitemoehybrid":
        model_type = "granitehybrid"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type=model_type, **kwargs
    )


_patch_granitehybrid_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
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
    """Available Granite 4.0 H-Small GGUF model variants for causal language modeling."""

    GRANITE_4_0_H_SMALL_GGUF = "H_SMALL_GGUF"


class ModelLoader(ForgeModel):
    """Granite 4.0 H-Small GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_4_0_H_SMALL_GGUF: LLMModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-h-small-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_4_0_H_SMALL_GGUF

    GGUF_FILE = "granite-4.0-h-small-Q4_K_M.gguf"

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
            model="Granite 4.0 H-Small GGUF",
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
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
