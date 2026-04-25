# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zen4 Thinking GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

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


def _patch_transformers_qwen3next_gguf():
    """Monkey-patch transformers to add qwen3next GGUF architecture support.

    Transformers 5.x has Qwen3NextForCausalLM but lacks GGUF loading support
    for the qwen3next architecture. We bridge the gap by registering qwen3next
    config/tensor mappings and converting the model_type to qwen3_next.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import sys

    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register qwen3next as a supported GGUF architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    # 2. Add config mapping for qwen3next → Qwen3NextConfig fields
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",  # post-processed to rope_parameters dict
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
        "ssm.time_step_rank": "linear_num_value_heads",
        "full_attention_interval": "full_attention_interval",
    }

    # 3. Reuse Qwen2MoeTensorProcessor for qwen3next (handles combined gate_up MoE experts)
    TENSOR_PROCESSORS["qwen3next"] = Qwen2MoeTensorProcessor

    # 4. Register tokenizer converter (qwen3_next → qwen2_moe per conversion_mapping.py)
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen2_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3next"] = GGUF_TO_FAST_CONVERTERS["qwen2_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_next"] = GGUF_TO_FAST_CONVERTERS["qwen2_moe"]

    # 5. Patch load_gguf_checkpoint to remap model_type and fix rope/layer_types config
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen3next":
            config = result["config"]
            config["model_type"] = "qwen3_next"

            # Convert flat rope_theta to rope_parameters dict
            rope_theta = config.pop("rope_theta", None)
            if rope_theta is not None:
                config["rope_parameters"] = {
                    "rope_theta": float(rope_theta),
                    "partial_rotary_factor": 0.25,
                    "rope_type": "default",
                }

            # Convert full_attention_interval to layer_types list
            num_layers = config.get("num_hidden_layers", 48)
            interval = config.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            config["layer_types"] = layer_types

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to remap qwen3_next → qwen3moe for the tensor name map.
    # The qwen3next name map has a combined ffn_gate_up_exps entry, but the actual GGUF file
    # stores separate ffn_gate_exps + ffn_up_exps (qwen3moe format). Using qwen3moe triggers
    # the perform_fallback_tensor_mapping path that recombines them into gate_up_proj.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_next", "qwen3next"):
            model_type = "qwen3moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map

    # 7. Patch is_gguf_available to handle 'N/A' version strings
    _orig_is_gguf_available = gguf_utils.is_gguf_available

    def _patched_is_gguf_available(*args, **kwargs):
        try:
            return _orig_is_gguf_available(*args, **kwargs)
        except Exception:
            try:
                import importlib.metadata
                from packaging.version import Version

                gguf_ver = importlib.metadata.version("gguf")
                min_ver = args[0] if args else kwargs.get("min_version", "0.10.0")
                return Version(gguf_ver) >= Version(min_ver)
            except Exception:
                return False

    gguf_utils.is_gguf_available = _patched_is_gguf_available


# Apply the monkey-patch at import time
_patch_transformers_qwen3next_gguf()


class ModelVariant(StrEnum):
    """Available Zen4 Thinking GGUF model variants for causal language modeling."""

    ZEN4_THINKING_GGUF = "GGUF"


class ModelLoader(ForgeModel):
    """Zen4 Thinking GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.ZEN4_THINKING_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/zen4-thinking-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZEN4_THINKING_GGUF

    GGUF_FILE = "zen4-thinking.Q4_K_M.gguf"

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
            model="Zen4 Thinking GGUF",
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
