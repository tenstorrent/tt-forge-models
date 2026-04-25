# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
njokukelechi Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model loader
implementation for causal language modeling.
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
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    already_registered = "qwen35moe" in GGUF_SUPPORTED_ARCHITECTURES

    if not already_registered:
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

    # 3. Reuse qwen3moe tensor processor for qwen35moe
    if "qwen3moe" in TENSOR_PROCESSORS:
        TENSOR_PROCESSORS["qwen35moe"] = TENSOR_PROCESSORS["qwen3moe"]

    # 4. Register tokenizer converter
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen35moe"] = GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        GGUF_TO_FAST_CONVERTERS["qwen3_5_moe_text"] = GGUF_TO_FAST_CONVERTERS[
            "qwen3_moe"
        ]

    # 5. Patch load_gguf_checkpoint to handle qwen35moe -> qwen3_5_moe_text
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
            # GGUF stores conv1d.weight as 2D [out, kernel] but HF Conv1d
            # expects 3D [out, in/groups, kernel]; insert the groups dim.
            if "tensors" in result:
                import torch as _torch

                for key in list(result["tensors"].keys()):
                    if key.endswith("conv1d.weight"):
                        t = result["tensors"][key]
                        if t.ndim == 2:
                            result["tensors"][key] = t.unsqueeze(1)
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # 6. Patch get_gguf_hf_weights_map to force qwen35moe arch and add
    # ffn_gate_exps/ffn_up_exps -> gate_up_proj alias mappings.
    #
    # qwen35moe name_map maps gate_up_proj -> ffn_gate_up_exps; other patches
    # (e.g. 4_5test_gguf) may remap qwen3_5_moe_text -> qwen3moe which lacks
    # that entry.  We intercept first, recover hf_model from any thread-local
    # that an inner patch stored it in (e.g. 4_5test_gguf pops model_to_load),
    # then call orig with model_type="qwen35moe" so the correct name_map is
    # used, and finally add the separate-gate-tensor aliases.
    import re as _re
    import sys as _sys

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if hf_model is None:
            # An inner patch (e.g. 4_5test_gguf) may have saved model_to_load
            # in a module-level thread-local before popping it from kwargs.
            for _mod in _sys.modules.values():
                _ctx = getattr(_mod, "_model_to_load_ctx", None)
                if _ctx is not None:
                    _recovered = getattr(_ctx, "model_to_load", None)
                    if _recovered is not None:
                        hf_model = _recovered
                        break
        if model_type is None and hf_model is not None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe", "qwen35moe"):
            # Force qwen35moe so gate_up_proj -> ffn_gate_up_exps is in the map.
            # Passing this explicit type bypasses inner patches that would map
            # qwen3_5_moe_text -> qwen3moe (which lacks ffn_gate_up_exps).
            model_type = "qwen35moe"
        result = orig_get_map(hf_model, processor, model_type, num_layers, qual_name)
        if model_type == "qwen35moe":
            # Build ffn_gate_exps/ffn_up_exps -> gate_up_proj alias mappings.
            # The fused ffn_gate_up_exps entry is in result; derive the
            # per-tensor entries so the MoE processor can interleave them.
            gate_up_entries = {
                k: v
                for k, v in result.items()
                if _re.search(r"blk\.\d+\.ffn_gate_up_exps", k)
            }
            for fused_key, hf_name in gate_up_entries.items():
                gate_key = fused_key.replace("ffn_gate_up_exps", "ffn_gate_exps")
                up_key = fused_key.replace("ffn_gate_up_exps", "ffn_up_exps")
                result.setdefault(gate_key, hf_name)
                result.setdefault(up_key, hf_name)
        return result

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


# Apply the monkey-patch at import time
_patch_transformers_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available njokukelechi Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model variants."""

    QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_Q4_K_M_GGUF = (
        "35B_A3B_Uncensored_HauhauCS_Aggressive_Q4_K_M_GGUF"
    )


class ModelLoader(ForgeModel):
    """njokukelechi Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="njokukelechi/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.QWEN_3_5_35B_A3B_UNCENSORED_HAUHAUCS_AGGRESSIVE_Q4_K_M_GGUF
    )

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
            model="njokukelechi Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive GGUF",
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
                if hasattr(config, "layer_types") and config.layer_types is not None:
                    config.layer_types = config.layer_types[: self.num_layers]
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
