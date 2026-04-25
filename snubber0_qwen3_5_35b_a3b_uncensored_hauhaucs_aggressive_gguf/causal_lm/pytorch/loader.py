# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Snubber0 Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model loader for causal language modeling.
"""

from typing import Optional

import torch
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
    """Monkey-patch transformers to add qwen35moe GGUF architecture support.

    Transformers 5.x has Qwen3_5MoeForCausalLM but lacks GGUF loading support
    for the qwen35moe architecture. We bridge the gap by registering qwen35moe
    config/tensor mappings and converting the model_type to qwen3_5_moe_text.
    """
    import re

    import numpy as np
    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        GGUFTensor,
        Qwen2MoeTensorProcessor,
        TENSOR_PROCESSORS,
    )

    # Custom tensor processor for qwen35moe.
    # Qwen3.5MoE stores fused gate_up_proj expert weights in HF format, but the
    # GGUF file names tensors as blk.N.ffn_gate_exps.weight (WITH .weight suffix).
    # Qwen2MoeTensorProcessor.process captures m["name"] (WITHOUT .weight) and
    # looks it up in tensor_key_mapping, but the fallback mapping adds keys WITH
    # .weight — a mismatch. We fix process() to look up with the .weight suffix.
    class Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor):
        def process(self, weights, name: str, **kwargs):
            if m := re.fullmatch(self.GGUF_MOE_WEIGHTS_PATTERN, name):
                tensor_key_mapping = kwargs.get("tensor_key_mapping")
                parsed_parameters = kwargs.get("parsed_parameters")
                if tensor_key_mapping:
                    # GGUF tensors for this model have .weight suffix; the fallback
                    # mapping stores keys WITH .weight, but the parent looks up
                    # m["name"] (without .weight).  Try both.
                    key = m["name"] + ".weight"
                    if key not in tensor_key_mapping:
                        key = m["name"]
                    if key in tensor_key_mapping:
                        self._set_moe_expert_tensor(
                            weights, parsed_parameters, tensor_key_mapping[key], m["w"]
                        )
                        return GGUFTensor(weights, None, {})
            if "ffn_gate_inp_shexp" in name:
                weights = np.expand_dims(weights, axis=0)
            return GGUFTensor(weights, name, {})

    # Always install our custom processor (overrides any coarser prior registration).
    TENSOR_PROCESSORS["qwen35moe"] = Qwen35MoeTensorProcessor

    already_patched = "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES

    if not already_patched:
        # 1. Register qwen35moe as a supported architecture
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

        # 2. Add config mapping for qwen35moe (based on qwen3_moe + Qwen3.5 fields)
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

        # 3. Register tokenizer converter
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

        if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
            GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
            GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
                "qwen3_moe"
            ]

        # 4. Patch load_gguf_checkpoint to handle qwen35moe -> qwen3_5_moe_text
        orig_load = gguf_utils.load_gguf_checkpoint

        def patched_load_gguf_checkpoint(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            if result.get("config", {}).get("model_type") == "qwen35moe":
                result["config"]["model_type"] = "qwen3_5_moe_text"
                # Generate layer_types from full_attention_interval
                config = result["config"]
                num_layers = config.get("num_hidden_layers", 40)
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

        # Also patch modules that imported load_gguf_checkpoint directly
        import transformers.configuration_utils as config_utils
        import transformers.modeling_utils as modeling_utils
        import transformers.models.auto.tokenization_auto as tok_auto

        for mod in (tok_auto, config_utils, modeling_utils):
            if hasattr(mod, "load_gguf_checkpoint"):
                mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Always re-apply the get_gguf_hf_weights_map patch to ensure we use qwen3moe
    # (not qwen35moe) for the name map.  Another loader (onion008) may have patched
    # this already but mapped to qwen35moe, which produces wrong tensor key names.
    # We re-patch unconditionally so the correct mapping is always last in the chain.
    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe", "qwen35moe"):
            model_type = "qwen3moe"
        return orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Snubber0 Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model variants."""

    QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_GGUF = (
        "35B_A3B_Uncensored_HauhauCS_Aggressive_GGUF"
    )


class ModelLoader(ForgeModel):
    """Snubber0 Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_GGUF: LLMModelConfig(
            pretrained_model_name="Snubber0/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

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
            model="Snubber0 Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF",
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
