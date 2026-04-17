# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored i1 GGUF model loader implementation for causal language modeling.
"""
import re

import numpy as np
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
    for the qwen3next architecture. The gguf library (>=0.18) already knows about
    qwen3next tensor names, so we only need to bridge transformers' config/tensor
    processing layer.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
        TENSOR_PROCESSORS,
        GGUFTensor,
        TensorProcessor,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3next" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3next")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3next"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": "moe_intermediate_size",
        "expert_shared_feed_forward_length": "shared_expert_intermediate_size",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.state_size": None,
        "ssm.inner_size": None,
        "ssm.time_step_rank": None,
        "ssm.group_count": None,
    }

    GGUF_MOE_WEIGHTS_RE = re.compile(
        r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)\.weight$"
    )
    GGUF_QKVZ_RE = re.compile(
        r"(?P<name>blk\.(?P<bid>\d+)\.(?P<part>attn_qkv|attn_gate))\.weight$"
    )

    class Qwen3NextTensorProcessor(TensorProcessor):
        def __init__(self, config=None):
            super().__init__(config=config)

        def preprocess_name(self, hf_name: str) -> str:
            hf_name = hf_name.replace(".dt_bias", ".dt_proj")
            return hf_name

        def process(self, weights, name: str, **kwargs):
            tensor_key_mapping = kwargs.get("tensor_key_mapping")
            parsed_parameters = kwargs.get("parsed_parameters")

            if m := re.fullmatch(GGUF_MOE_WEIGHTS_RE, name):
                if tensor_key_mapping and m["name"] in tensor_key_mapping:
                    self._set_moe_expert_tensor(
                        weights,
                        parsed_parameters,
                        tensor_key_mapping[m["name"]],
                        m["w"],
                    )
                    return GGUFTensor(weights, None, {})

            if m := re.fullmatch(GGUF_QKVZ_RE, name):
                if tensor_key_mapping and m["name"] in tensor_key_mapping:
                    self._merge_qkvz_tensor(
                        weights,
                        parsed_parameters,
                        tensor_key_mapping[m["name"]],
                        m["part"],
                    )
                    return GGUFTensor(weights, None, {})

            if "ffn_gate_inp_shexp" in name:
                weights = np.expand_dims(weights, axis=0)

            if "ssm_conv1d" in name:
                if weights.ndim == 2:
                    weights = np.transpose(weights)
                    weights = np.expand_dims(weights, axis=1)

            if "ssm_a" in name:
                weights = np.log(-weights)

            return GGUFTensor(weights, name, {})

        def _set_moe_expert_tensor(
            self,
            weights: np.ndarray,
            parsed_parameters: dict,
            hf_name: str,
            w: str,
        ):
            torch_weights = torch.from_numpy(np.copy(weights))
            if w == "down":
                parsed_parameters["tensors"][hf_name] = torch_weights
            else:
                shape = list(weights.shape)
                shard_dim = 1
                shard_size = shape[shard_dim]
                shape[shard_dim] = shard_size * 2
                if hf_name not in parsed_parameters["tensors"]:
                    parsed_parameters["tensors"][hf_name] = torch.zeros(
                        shape, dtype=torch_weights.dtype
                    )
                out = parsed_parameters["tensors"][hf_name]
                if w == "gate":
                    out = out.narrow(shard_dim, 0, shard_size)
                else:
                    out = out.narrow(shard_dim, shard_size, shard_size)
                out.copy_(torch_weights)

        def _merge_qkvz_tensor(
            self,
            weights: np.ndarray,
            parsed_parameters: dict,
            hf_name: str,
            part: str,
        ):
            torch_weights = torch.from_numpy(np.copy(weights))
            if hf_name not in parsed_parameters["tensors"]:
                parsed_parameters["tensors"][hf_name] = torch_weights
            else:
                existing = parsed_parameters["tensors"][hf_name]
                if part == "attn_gate":
                    parsed_parameters["tensors"][hf_name] = torch.cat(
                        [existing, torch_weights], dim=0
                    )
                else:
                    parsed_parameters["tensors"][hf_name] = torch.cat(
                        [torch_weights, existing], dim=0
                    )

    TENSOR_PROCESSORS["qwen3next"] = Qwen3NextTensorProcessor

    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "qwen3next" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen3next"] = GGUFQwen2Converter

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen3next":
            config["model_type"] = "qwen3_next"

            num_layers = config.get("num_hidden_layers", 48)
            full_attention_interval = 4
            layer_types = []
            for i in range(num_layers):
                if (i + 1) % full_attention_interval == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("linear_attention")
            config["layer_types"] = layer_types

            head_dim = config.get("head_dim", 256)
            rope_dim = config.pop("rope.dimension_count", None)
            if rope_dim is not None and head_dim > 0:
                config["partial_rotary_factor"] = rope_dim / head_dim

            ssm_state_size = config.pop("ssm.state_size", None)
            ssm_group_count = config.pop("ssm.group_count", None)
            ssm_time_step_rank = config.pop("ssm.time_step_rank", None)
            ssm_inner_size = config.pop("ssm.inner_size", None)

            if ssm_state_size is not None:
                config["linear_key_head_dim"] = ssm_state_size
            if ssm_group_count is not None:
                config["linear_num_key_heads"] = ssm_group_count
            if ssm_time_step_rank is not None:
                config["linear_num_value_heads"] = ssm_time_step_rank
            if ssm_inner_size is not None and ssm_time_step_rank:
                config["linear_value_head_dim"] = ssm_inner_size // ssm_time_step_rank

        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint

    orig_get_map = gguf_utils.get_gguf_hf_weights_map

    def patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type == "qwen3_next":
            model_type = "qwen3next"
        result = orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

        for key in list(result.keys()):
            m = re.match(r"(blk\.\d+)\.ffn_gate_up_exps$", key)
            if m:
                result[f"{m.group(1)}.ffn_gate_exps"] = result[key]
                result[f"{m.group(1)}.ffn_up_exps"] = result[key]

            m = re.match(r"(blk\.\d+)\.ssm_in(\.weight)?$", key)
            if m:
                suffix = m.group(2) or ""
                result[f"{m.group(1)}.attn_qkv{suffix}"] = result[key]
                result[f"{m.group(1)}.attn_gate{suffix}"] = result[key]

        return result

    gguf_utils.get_gguf_hf_weights_map = patched_get_gguf_hf_weights_map


_patch_transformers_qwen3next_gguf()


class ModelVariant(StrEnum):
    """Available Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored i1 GGUF model variants for causal language modeling."""

    QWEN3_NEXT_80B_A3B_THINKING_GRPO_UNCENSORED_I1_GGUF = (
        "80B_A3B_Thinking_GRPO_Uncensored_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored i1 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_NEXT_80B_A3B_THINKING_GRPO_UNCENSORED_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_NEXT_80B_A3B_THINKING_GRPO_UNCENSORED_I1_GGUF

    GGUF_FILE = "Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored.i1-Q4_K_M.gguf"

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
            model="Qwen3-Next-80B-A3B-Thinking-GRPO-Uncensored i1 GGUF",
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
            if hasattr(config, "layer_types"):
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
