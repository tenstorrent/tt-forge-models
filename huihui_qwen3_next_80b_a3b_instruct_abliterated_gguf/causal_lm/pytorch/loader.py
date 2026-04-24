# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Huihui Qwen3-Next-80B-A3B-Instruct abliterated GGUF model loader implementation for causal language modeling.
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
    Qwen2MoeTensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

# Qwen3-Next-80B-A3B is a hybrid SSM+MoE architecture (linear_attention + full_attention
# layers). The GGUF file declares architecture as 'qwen3next'. Transformers 5.x has
# Qwen3NextForCausalLM (model_type="qwen3_next") but no GGUF loading support for it.
#
# Config field mapping for qwen3next GGUF → Qwen3NextConfig:
_QWEN3NEXT_CONFIG_MAPPING = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,  # used for partial_rotary_factor=0.25 (64/256)
    "rope.freq_base": "rope_theta",  # moved into rope_parameters in post-processing
    "attention.key_length": "head_dim",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    "expert_count": "num_experts",
    "expert_used_count": "num_experts_per_tok",
    "expert_feed_forward_length": "moe_intermediate_size",
    "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.state_size": "linear_key_head_dim",
    "ssm.group_count": "linear_num_key_heads",
}


def _patch_qwen3next_support():
    """Register qwen3next GGUF architecture mapped to the qwen3_next transformers model.

    gguf-py 0.18+ already knows about qwen3next tensor names. Transformers 5.x has
    Qwen3NextForCausalLM but lacks GGUF config/tokenizer registration for it.
    """
    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
        "qwen3next"
    ] = _QWEN3NEXT_CONFIG_MAPPING

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3next"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]

    if hasattr(_gguf_utils, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        _gguf_utils.GGUF_CONFIG_DEFAULTS_MAPPING["qwen3next"] = {
            "norm_topk_prob": True,
        }

    # Use Qwen2MoeTensorProcessor to merge split gate/up expert weights into
    # gate_up_proj, matching the Qwen3Next MoE block layout.
    _gguf_utils.TENSOR_PROCESSORS["qwen3next"] = Qwen2MoeTensorProcessor


def _patched_load_gguf_checkpoint(*args, **kwargs):
    """Wrap load_gguf_checkpoint to add qwen3next support and build rope_parameters."""
    _patch_qwen3next_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    config = result.get("config", {})
    if config.get("model_type") == "qwen3next":
        config["model_type"] = "qwen3_next"
        # Qwen3NextConfig uses rope_parameters dict instead of a flat rope_theta.
        # partial_rotary_factor = rope.dimension_count / head_dim = 64 / 256 = 0.25
        rope_theta = config.pop("rope_theta", 10_000_000.0)
        config["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": 0.25,
        }
        # linear_value_head_dim mirrors linear_key_head_dim (both 128 in this model)
        if "linear_key_head_dim" in config:
            config.setdefault("linear_value_head_dim", config["linear_key_head_dim"])
    return result


def _patched_get_gguf_hf_weights_map(hf_model, processor, model_type=None, **kwargs):
    """Wrap get_gguf_hf_weights_map to map qwen3_next HF model → qwen3next gguf-py arch."""
    if model_type is None and hasattr(hf_model, "config"):
        model_type = hf_model.config.model_type
    if model_type == "qwen3_next":
        model_type = "qwen3next"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type=model_type, **kwargs
    )


_patch_qwen3next_support()
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
    """Available Huihui Qwen3-Next-80B-A3B-Instruct abliterated GGUF model variants for causal language modeling."""

    HUIHUI_QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF = (
        "80B_A3B_Instruct_abliterated_GGUF"
    )


class ModelLoader(ForgeModel):
    """Huihui Qwen3-Next-80B-A3B-Instruct abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF

    _GGUF_FILES = {
        ModelVariant.HUIHUI_QWEN_3_NEXT_80B_A3B_INSTRUCT_ABLITERATED_GGUF: "Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated.Q4_K_M.gguf",
    }

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Huihui Qwen3-Next-80B-A3B-Instruct abliterated GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._gguf_file

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
        model_kwargs["gguf_file"] = self._gguf_file

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._gguf_file
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
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
