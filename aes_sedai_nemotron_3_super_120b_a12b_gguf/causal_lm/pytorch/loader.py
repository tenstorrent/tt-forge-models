# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling.
"""
import sys
from typing import Optional

import torch
import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
    NemotronTensorProcessor,
    TENSOR_PROCESSORS,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUF_CONFIG_MAPPING

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

# NemotronH layer pattern for NVIDIA-Nemotron-3-Super-120B-A12B
_HYBRID_OVERRIDE_PATTERN = (
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
    "EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
)

# NemotronHConfig field mappings from GGUF nemotron_h_moe metadata
_NEMOTRON_H_MOE_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
    "vocab_size": "vocab_size",
    "ssm.conv_kernel": "conv_kernel",
    "ssm.state_size": "ssm_state_size",
    "ssm.group_count": "n_groups",
    "expert_used_count": "num_experts_per_tok",
    "expert_group_count": "n_group",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "moe_shared_expert_intermediate_size",
    "expert_count": "n_routed_experts",
    "expert_shared_count": "n_shared_experts",
    "moe_latent_size": "moe_latent_size",
}


def _patch_nemotron_h_moe_support():
    """Register nemotron_h_moe GGUF architecture and NemotronHConfig."""
    if "nemotron_h_moe" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # Register nemotron_h_moe in GGUF supported architectures
    GGUF_SUPPORTED_ARCHITECTURES.append("nemotron_h_moe")

    # Add config field mappings
    GGUF_CONFIG_MAPPING["nemotron_h_moe"] = _NEMOTRON_H_MOE_CONFIG_MAPPING
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if section == "config":
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "nemotron_h_moe"
            ] = _NEMOTRON_H_MOE_CONFIG_MAPPING

    # Reuse nemotron tensor processor (handles norm.weight offset)
    TENSOR_PROCESSORS["nemotron_h_moe"] = NemotronTensorProcessor

    # Reuse nemotron fast tokenizer converter if available
    if "nemotron" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["nemotron_h_moe"] = GGUF_TO_FAST_CONVERTERS["nemotron"]

    # Register NemotronHConfig from HEAD transformers (installed in /tmp)
    _register_nemotron_h_config()


def _register_nemotron_h_config():
    """Register NemotronHConfig in the auto config mapping."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "nemotron_h" in CONFIG_MAPPING:
        return

    # Try loading NemotronHConfig from HEAD transformers in /tmp
    _head_path = "/tmp/transformers_site_packages"
    if _head_path not in sys.path:
        sys.path.insert(0, _head_path)

    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            NemotronHForCausalLM,
        )
    except ImportError:
        return

    CONFIG_MAPPING["nemotron_h"] = NemotronHConfig

    # Register in model auto mappings
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["nemotron_h"] = "NemotronHForCausalLM"
    except (ImportError, AttributeError):
        pass


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add nemotron_h_moe support."""
    _patch_nemotron_h_moe_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    cfg = result.get("config", {})
    if cfg.get("model_type") == "nemotron_h_moe":
        cfg["model_type"] = "nemotron_h"
        cfg["hybrid_override_pattern"] = _HYBRID_OVERRIDE_PATTERN
        # GGUF reports 0 KV heads for MHA layers; use actual value from config
        if cfg.get("num_key_value_heads", 0) == 0:
            cfg["num_key_value_heads"] = 2
    return result


_patch_nemotron_h_moe_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint


class ModelVariant(StrEnum):
    """Available AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model variants for causal language modeling."""

    AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF = "AesSedai_3_Super_120B_A12B_GGUF"


class ModelLoader(ForgeModel):
    """AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AES_SEDAI_NEMOTRON_3_SUPER_120B_A12B_GGUF

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
            model="AesSedai NVIDIA Nemotron 3 Super 120B A12B GGUF",
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                if hasattr(mlp, "experts"):
                    shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                    shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
                if hasattr(mlp, "shared_expert"):
                    shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                    shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                    shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")
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
