# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling.
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
    TensorProcessor,
    GGUFTensor,
)

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

# hybrid_override_pattern for the 120B model (88 layers):
# M=mamba, E=moe, *=attention
_NEMOTRON_H_MOE_HYBRID_PATTERN = (
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
    "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
)

# GGUF key -> NemotronHConfig field mapping
_NEMOTRON_H_MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "attention.key_length": "head_dim",
    "vocab_size": "vocab_size",
    "expert_used_count": "num_experts_per_tok",
    "expert_group_count": "n_group",
    "expert_group_used_count": "topk_group",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
    "expert_count": "n_routed_experts",
    "expert_shared_count": "n_shared_experts",
    "moe_latent_size": "moe_latent_size",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.state_size": "ssm_state_size",
    "ssm.group_count": "n_groups",
}


class _NemotronHMoeTensorProcessor(TensorProcessor):
    """Tensor processor for nemotron_h_moe GGUF files.

    gguf-py uses backbone.layers.{bid}.mixer.* for NemotronH tensors while
    transformers uses model.layers.{bid}.mixer.*. Remapping enables the
    standard name_map lookup to resolve most tensors correctly.
    """

    def preprocess_name(self, hf_name: str) -> str:
        return (
            hf_name.replace("model.embeddings", "backbone.embeddings")
            .replace("model.layers.", "backbone.layers.")
            .replace("model.norm_f", "backbone.norm_f")
        )

    def process(self, weights, name, **kwargs):
        return GGUFTensor(weights, name, {})


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe as a supported GGUF architecture."""
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        mapping = _NEMOTRON_H_MOE_CONFIG_MAPPING if section == "config" else {}
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["nemotron_h_moe"] = mapping

    # Register tokenizer converter (gpt2-style tokenizer).
    # Both the raw GGUF architecture name and the transformed model_type must be
    # registered because tokenization_utils_tokenizers reads model_type from the
    # patched config (which is already 'nemotron_h').
    try:
        import transformers.integrations.ggml as _ggml

        if (
            hasattr(_ggml, "GGUF_TO_FAST_CONVERTERS")
            and "gpt2" in _ggml.GGUF_TO_FAST_CONVERTERS
        ):
            for arch in ("nemotron_h_moe", "nemotron_h"):
                _ggml.GGUF_TO_FAST_CONVERTERS[arch] = _ggml.GGUF_TO_FAST_CONVERTERS[
                    "gpt2"
                ]
    except Exception:
        pass

    _gguf_utils.TENSOR_PROCESSORS["nemotron_h_moe"] = _NemotronHMoeTensorProcessor


def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    """Wrap load_gguf_checkpoint to support the nemotron_h_moe GGUF architecture."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, **kwargs
    )
    cfg = result.get("config", {})
    if cfg.get("model_type") == "nemotron_h_moe":
        cfg["model_type"] = "nemotron_h"
        cfg["hybrid_override_pattern"] = _NEMOTRON_H_MOE_HYBRID_PATTERN
        cfg["architectures"] = ["NemotronHForCausalLM"]
        # num_key_value_heads is a per-layer array in the GGUF; use the max non-zero value
        kv_heads = cfg.get("num_key_value_heads")
        if isinstance(kv_heads, list):
            cfg["num_key_value_heads"] = max((v for v in kv_heads if v > 0), default=2)
    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available Nemotron 3 Super 120B GGUF model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF = (
        "3_Super_120B_A12B_BF16_heretic_i1_GGUF"
    )
    GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF = "ggml_org_3_Super_120B_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron 3 Super 120B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: LLMModelConfig(
            pretrained_model_name="ggml-org/Nemotron-3-Super-120B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_BF16_HERETIC_I1_GGUF: "NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic.i1-Q4_K_M.gguf",
        ModelVariant.GGML_ORG_NEMOTRON_3_SUPER_120B_GGUF: "Nemotron-3-Super-120B-Q4_K.gguf",
    }

    sample_text = "Give me a short introduction to large language model."

    @property
    def gguf_file(self):
        return self._GGUF_FILES[self._variant]

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
            model="Nemotron 3 Super 120B GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.gguf_file
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
        )
        return self.config
