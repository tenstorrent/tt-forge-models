# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
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
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUF_CONFIG_MAPPING

_HYBRID_OVERRIDE_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
_PATTERN_MAPPING = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}

_NEMOTRON_H_MOE_CONFIG_MAPPING = {
    "block_count": "num_hidden_layers",
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "feed_forward_length": None,
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "rope.freq_base": "rope_theta",
    "rope.dimension_count": None,
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "attention.key_length": "head_dim",
    "vocab_size": "vocab_size",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.state_size": "ssm_state_size",
    "ssm.group_count": "n_groups",
    "expert_used_count": "num_experts_per_tok",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
    "expert_count": "n_routed_experts",
    "expert_shared_count": "n_shared_experts",
    "expert_weights_norm": "norm_topk_prob",
    "expert_weights_scale": "routed_scaling_factor",
    "moe_latent_size": "moe_latent_size",
}


def _patch_nemotron_h_moe_support():
    if "nemotron_h_moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "nemotron" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section].setdefault(
                "nemotron_h_moe",
                _NEMOTRON_H_MOE_CONFIG_MAPPING
                if section == "config"
                else _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron"],
            )
    GGUF_CONFIG_MAPPING.setdefault("nemotron_h_moe", _NEMOTRON_H_MOE_CONFIG_MAPPING)
    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "nemotron_h_moe", GGUF_TO_FAST_CONVERTERS["nemotron"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "nemotron_h", GGUF_TO_FAST_CONVERTERS["nemotron"]
        )


def _fix_nemotron_h_moe_config(result):
    if result.get("config", {}).get("model_type") == "nemotron_h_moe":
        result["config"]["model_type"] = "nemotron_h"
        num_kv = result["config"].get("num_key_value_heads")
        if isinstance(num_kv, list):
            result["config"]["num_key_value_heads"] = max(num_kv)
        elif not num_kv:
            result["config"]["num_key_value_heads"] = 2
        result["config"]["layers_block_type"] = [
            _PATTERN_MAPPING[c] for c in _HYBRID_OVERRIDE_PATTERN
        ]
        result["config"].setdefault(
            "intermediate_size", result["config"].get("moe_intermediate_size", 2688)
        )
    return result


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    _patch_nemotron_h_moe_support()
    if return_tensors:
        # When the GGUF has no tensors (header-only cache), skip the weight
        # loading chain to avoid get_gguf_hf_weights_map(None) crashes.
        import gguf as _gguf_lib

        reader = _gguf_lib.GGUFReader(gguf_path)
        if len(reader.tensors) == 0:
            result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=False)
            result.setdefault("tensors", {})
            return _fix_nemotron_h_moe_config(result)
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    return _fix_nemotron_h_moe_config(result)


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
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF = "3_Super_120B_A12B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_Q4_K_M_GGUF

    GGUF_FILE = (
        "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf"
    )

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
            model="Nemotron 3 Super 120B A12B AesSedai GGUF",
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

        # Re-install our patch immediately before from_pretrained so that
        # modeling_utils.py's lazy `from .modeling_gguf_pytorch_utils import
        # load_gguf_checkpoint` gets a version that accepts model_to_load.
        # Other GGUF loaders may have overwritten _gguf_utils.load_gguf_checkpoint
        # after our module-level patch with signatures that drop model_to_load.
        _prev_gguf_load = _gguf_utils.load_gguf_checkpoint
        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        finally:
            _gguf_utils.load_gguf_checkpoint = _prev_gguf_load

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
