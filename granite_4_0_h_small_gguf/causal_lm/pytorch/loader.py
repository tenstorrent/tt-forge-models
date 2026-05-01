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

# GGUF uses "granitehybrid" for GraniteMoeHybrid (ibm-granite/granite-4.0-h-small),
# which transformers 5.x does not yet recognise as a supported GGUF architecture.
_GRANITEHYBRID_CONFIG_KEYS = {
    "vocab_size": "vocab_size",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": "intermediate_size",
    "block_count": "num_hidden_layers",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "expert_count": "num_local_experts",
    "expert_used_count": "num_experts_per_tok",
    "attention.scale": "attention_multiplier",
    "embedding_scale": "embedding_multiplier",
    "residual_scale": "residual_multiplier",
    "logit_scale": "logits_scaling",
    "expert_shared_feed_forward_length": "shared_intermediate_size",
    "ssm.conv_kernel": "mamba_d_conv",
    "ssm.state_size": "mamba_d_state",
    "ssm.group_count": "mamba_n_groups",
    "ssm.inner_size": "_mamba_inner_size",
    "ssm.time_step_rank": "mamba_n_heads",
}


def _patch_granitehybrid_support():
    if "granitehybrid" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("granitehybrid")
    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].setdefault(
        "granitehybrid", _GRANITEHYBRID_CONFIG_KEYS
    )
    # granitehybrid uses a gpt2-style BPE tokenizer
    GGUF_TO_FAST_CONVERTERS.setdefault(
        "granitemoehybrid", GGUF_TO_FAST_CONVERTERS["gpt2"]
    )


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to register granitehybrid and fix config."""
    _patch_granitehybrid_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )

    config = result.get("config", {})
    if config.get("model_type") == "granitehybrid":
        # head_count_kv is a per-layer list (0=mamba, N=attention); derive layer_types
        kv_heads = config.get("num_key_value_heads", [])
        if isinstance(kv_heads, list):
            config["layer_types"] = [
                "attention" if h > 0 else "mamba" for h in kv_heads
            ]
            config["num_key_value_heads"] = max(
                (int(h) for h in kv_heads if h > 0), default=8
            )

        # Derive mamba_expand from ssm.inner_size / hidden_size
        hidden_size = config.get("hidden_size", 4096)
        inner_size = config.pop("_mamba_inner_size", None)
        if inner_size is not None:
            config["mamba_expand"] = int(inner_size) // int(hidden_size)

        config["model_type"] = "granitemoehybrid"
        config["architectures"] = ["GraniteMoeHybridForCausalLM"]

    return result


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    """Remap granitemoehybrid → granitehybrid for gguf-py tensor name lookup."""
    resolved = (
        model_type
        if model_type is not None
        else getattr(getattr(hf_model, "config", None), "model_type", None)
    )
    if resolved == "granitemoehybrid":
        model_type = "granitehybrid"
    return _orig_get_gguf_hf_weights_map(
        hf_model,
        processor,
        model_type=model_type,
        num_layers=num_layers,
        qual_name=qual_name,
    )


_patch_granitehybrid_support()
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
            if hasattr(config, "layer_types") and config.layer_types is not None:
                config.layer_types = config.layer_types[: self.num_layers]
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
