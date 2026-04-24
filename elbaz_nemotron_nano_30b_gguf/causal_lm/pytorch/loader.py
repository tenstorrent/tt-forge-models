# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Elbaz Nemotron Nano 30B GGUF model loader implementation for causal language modeling.
"""
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe architecture for GGUF loading.

    The Nemotron-H MoE GGUF files declare architecture as 'nemotron_h_moe'
    which transformers does not yet recognise in its GGUF loader. The model
    itself is supported as 'nemotron_h' since transformers 5.5.
    """
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "nemotron" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            base = dict(_gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron"])
            base["expert_count"] = "n_routed_experts"
            base["expert_used_count"] = "num_experts_per_tok"
            base["expert_feed_forward_length"] = "moe_intermediate_size"
            base[
                "expert_shared_feed_forward_length"
            ] = "moe_shared_expert_intermediate_size"
            base["expert_shared_count"] = "n_shared_experts"
            base["attention.layer_norm_rms_epsilon"] = "layer_norm_epsilon"
            base["attention.layer_norm_epsilon"] = "layer_norm_epsilon"
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron_h_moe"] = base
    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS["nemotron"]
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "nemotron_h", GGUF_TO_FAST_CONVERTERS["nemotron"]
        )
    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING.setdefault("nemotron_h_moe", {})


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    cfg = result.get("config", {})
    if cfg.get("model_type") == "nemotron_h_moe":
        cfg["model_type"] = "nemotron_h"
    if isinstance(cfg.get("num_key_value_heads"), list):
        non_zero = [v for v in cfg["num_key_value_heads"] if v > 0]
        cfg["num_key_value_heads"] = non_zero[0] if non_zero else 8
    if isinstance(cfg.get("intermediate_size"), list):
        non_zero = [v for v in cfg["intermediate_size"] if v > 0]
        cfg["intermediate_size"] = max(non_zero) if non_zero else 1856
    return result


_patch_nemotron_h_moe_support()
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
    """Available Elbaz Nemotron Nano 30B GGUF model variants for causal language modeling."""

    ELBAZ_NEMOTRON_NANO_30B_A3B_IQ4_XS_GGUF = "30B_A3B_IQ4_XS_GGUF"


class ModelLoader(ForgeModel):
    """Elbaz Nemotron Nano 30B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ELBAZ_NEMOTRON_NANO_30B_A3B_IQ4_XS_GGUF: LLMModelConfig(
            pretrained_model_name="Ex0bit/Elbaz-NVIDIA-Nemotron-3-Nano-30B-A3B-PRISM",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ELBAZ_NEMOTRON_NANO_30B_A3B_IQ4_XS_GGUF

    GGUF_FILE = "Elbaz-NVIDIA-Nemotron-3-Nano-30B-A3B-PRISM-IQ4_XS.gguf"

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
            model="Elbaz Nemotron Nano 30B GGUF",
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
