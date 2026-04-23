# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 35B A3B Heretic v2 eq v1 GGUF model loader implementation for causal language modeling.
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


def _patch_transformers_qwen35moe_gguf():
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. We bridge the gap by registering qwen35moe
    config/tensor mappings and converting the model_type to qwen3_5_moe_text.

    Additionally, gguf-py 0.18+ maps qwen35moe gate_up_proj to the combined
    ffn_gate_up_exps tensor name, but older GGUF files use separate ffn_gate_exps
    and ffn_up_exps tensors. We add aliases so both naming conventions work.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

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

        if "qwen3moe" in TENSOR_PROCESSORS:
            TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
                "qwen3_moe"
            ]

        _orig_load = gguf_utils.load_gguf_checkpoint

        def _patched_load_gguf_checkpoint(*args, **kwargs):
            result = _orig_load(*args, **kwargs)
            if result.get("config", {}).get("model_type") == "qwen35moe":
                result["config"]["model_type"] = "qwen3_5_moe_text"
                config = result["config"]
                num_layers = config.get("num_hidden_layers", 40)
                interval = config.pop("full_attention_interval", 4)
                config["layer_types"] = [
                    "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(num_layers)
                ]
            return result

        gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

        import transformers.models.auto.tokenization_auto as tok_auto
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils

        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Always patch get_gguf_hf_weights_map to add backward-compatible tensor name aliases.
    # gguf-py 0.18+ uses ffn_gate_up_exps (combined) for qwen35moe, but many GGUF files
    # use the older separate ffn_gate_exps + ffn_up_exps naming. Add aliases for both.
    _orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe"):
            model_type = "qwen35moe"
        mapping = _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)
        if model_type == "qwen35moe":
            extras = {}
            for key, val in mapping.items():
                if "ffn_gate_up_exps" in key:
                    extras[key.replace("ffn_gate_up_exps", "ffn_gate_exps")] = val
                    extras[key.replace("ffn_gate_up_exps", "ffn_up_exps")] = val
            mapping.update(extras)
        return mapping

    gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Qwen3.5 35B A3B Heretic v2 eq v1 GGUF model variants for causal language modeling."""

    QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_I1_GGUF = "35B_A3B_Heretic_v2_eq_v1_I1_GGUF"
    QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_GGUF = "35B_A3B_Heretic_v2_eq_v1_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3.5 35B A3B Heretic v2 eq v1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-35B-A3B-heretic-v2-eq-v1-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-35B-A3B-heretic-v2-eq-v1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_I1_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_I1_GGUF: "Qwen3.5-35B-A3B-heretic-v2-eq-v1.i1-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_V2_EQ_V1_GGUF: "Qwen3.5-35B-A3B-heretic-v2-eq-v1.Q4_K_M.gguf",
    }

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
            model="Qwen3.5 35B A3B Heretic v2 eq v1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

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
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
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
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config
