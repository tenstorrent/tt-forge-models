# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher Qwen3.5 35B A3B GGUF model loader implementation for causal language modeling.
"""

import re as _re
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _patch_transformers_qwen35moe_gguf():
    """Register qwen35moe GGUF architecture support and fix fused expert weight mapping.

    qwen35moe GGUF files store MoE expert weights as separate ffn_gate_exps /
    ffn_up_exps tensors, but Qwen3_5MoeForCausalLM uses a fused gate_up_proj
    nn.Parameter.  We add the missing aliases in get_gguf_hf_weights_map so
    Qwen2MoeTensorProcessor can interleave gate and up into gate_up_proj.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    # Register qwen35moe as a supported architecture.
    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    # Add config key mapping for qwen35moe.
    if "qwen35moe" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = {
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
            "expert_count": "num_experts",
            "expert_used_count": "num_experts_per_tok",
            "full_attention_interval": "full_attention_interval",
        }

    # Reuse qwen3moe tensor processor for qwen35moe.
    if "qwen35moe" not in TENSOR_PROCESSORS and "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

    # Register tokenizer converters.
    if (
        "qwen35moe" not in GGUF_TO_FAST_CONVERTERS
        and "qwen3_moe" in GGUF_TO_FAST_CONVERTERS
    ):
        GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
    if (
        "qwen3_5_moe_text" not in GGUF_TO_FAST_CONVERTERS
        and "qwen3_moe" in GGUF_TO_FAST_CONVERTERS
    ):
        GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # Patch load_gguf_checkpoint to convert qwen35moe -> qwen3_5_moe_text.
    if not getattr(_gguf_utils, "_qwen35moe_load_patched", False):
        _orig_load = _gguf_utils.load_gguf_checkpoint

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = _orig_load(*args, **kwargs)
            if result.get("config", {}).get("model_type") == "qwen35moe":
                result["config"]["model_type"] = "qwen3_5_moe_text"
                config = result["config"]
                num_layers = config.get("num_hidden_layers", 40)
                interval = config.pop("full_attention_interval", 4)
                layer_types = [
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(num_layers)
                ]
                config["layer_types"] = layer_types
            return result

        _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
        _gguf_utils._qwen35moe_load_patched = True

        import transformers.models.auto.tokenization_auto as _tok_auto
        import transformers.configuration_utils as _config_utils
        import transformers.modeling_utils as _modeling_utils

        for _mod in (_tok_auto, _config_utils, _modeling_utils):
            if hasattr(_mod, "load_gguf_checkpoint"):
                _mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Patch get_gguf_hf_weights_map to add ffn_gate_exps/ffn_up_exps aliases.
    # qwen35moe GGUF has separate gate/up expert tensors, HF uses fused gate_up_proj.
    # The existing mapping only has blk.N.ffn_gate_up_exps; we alias the separate
    # GGUF names to the same fused HF parameter so Qwen2MoeTensorProcessor works.
    if not getattr(_gguf_utils, "_qwen35moe_weights_map_patched", False):
        _orig_get_map = _gguf_utils.get_gguf_hf_weights_map
        _fused_pattern = _re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps$")

        def _patched_get_gguf_hf_weights_map(
            hf_model, processor, model_type=None, num_layers=None, qual_name=""
        ):
            if model_type is None:
                model_type = hf_model.config.model_type
            if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
                model_type = "qwen35moe"
            result = _orig_get_map(
                hf_model, processor, model_type, num_layers, qual_name
            )
            if model_type == "qwen35moe":
                aliases = {}
                for key, val in result.items():
                    if m := _fused_pattern.match(key):
                        bid = m.group(1)
                        aliases[f"blk.{bid}.ffn_gate_exps"] = val
                        aliases[f"blk.{bid}.ffn_up_exps"] = val
                result.update(aliases)
            return result

        _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map
        _gguf_utils._qwen35moe_weights_map_patched = True


_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available mradermacher Qwen3.5 35B A3B GGUF model variants."""

    QWEN3_5_35B_A3B_Q4_K_M_GGUF = "35B_A3B_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher Qwen3.5 35B A3B GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_35B_A3B_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-35B-A3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_35B_A3B_Q4_K_M_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B.Q4_K_M.gguf"

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
            model="mradermacher Qwen3.5 35B A3B GGUF",
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
