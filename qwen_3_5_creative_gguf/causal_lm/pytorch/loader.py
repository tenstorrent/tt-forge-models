# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 Creative GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    TENSOR_PROCESSORS,
    Qwen2MoeTensorProcessor,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

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


def _patch_qwen35moe_gguf():
    """Patch transformers to support qwen35moe GGUF architecture.

    qwen35moe GGUF files store MoE experts as separate ffn_gate_exps and
    ffn_up_exps tensors, but the HF Qwen3_5MoeExperts module uses a fused
    gate_up_proj parameter. The Qwen2MoeTensorProcessor handles interleaving
    these into the fused tensor, but only when the tensor_key_mapping contains
    ffn_gate_exps/ffn_up_exps as keys pointing to gate_up_proj.

    The gguf-py qwen35moe architecture maps gate_up_proj -> ffn_gate_up_exps
    (a direct mapping), which prevents the fallback that would add the
    ffn_gate_exps/ffn_up_exps -> gate_up_proj mappings. We fix this by using
    qwen3moe (which has no direct gate_up_proj mapping) when building the
    tensor key mapping, triggering the fallback correctly.
    """
    if "qwen35moe_creative_patched" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe_creative_patched")

    if "qwen35moe" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen35moe")

    if "qwen35moe" not in GGUF_TO_TRANSFORMERS_MAPPING.get("config", {}):
        qwen3moe_config = GGUF_TO_TRANSFORMERS_MAPPING["config"].get("qwen3moe", {})
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"] = dict(qwen3moe_config)
        GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35moe"][
            "full_attention_interval"
        ] = "full_attention_interval"

    TENSOR_PROCESSORS["qwen35moe"] = Qwen2MoeTensorProcessor

    if "qwen3_moe" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen35moe", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )
        GGUF_TO_FAST_CONVERTERS.setdefault(
            "qwen3_5_moe_text", GGUF_TO_FAST_CONVERTERS["qwen3_moe"]
        )

    _orig_load = _gguf_utils.load_gguf_checkpoint

    def _patched_load_gguf_checkpoint(*args, **kwargs):
        result = _orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen35moe":
            result["config"]["model_type"] = "qwen3_5_moe_text"
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

    _gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
    for mod in (_config_utils, _auto_tokenizer, _tok_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = _patched_load_gguf_checkpoint

    # Fix get_gguf_hf_weights_map to use qwen3moe architecture for name lookup.
    # qwen35moe maps gate_up_proj -> ffn_gate_up_exps (direct), which prevents
    # the fallback that adds ffn_gate_exps/ffn_up_exps -> gate_up_proj mappings.
    # Using qwen3moe (where gate_up_proj -> None) triggers the fallback correctly.
    _orig_get_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe_text", "qwen3_5_moe", "qwen35moe"):
            model_type = "qwen3moe"
        return _orig_get_map(hf_model, processor, model_type, num_layers, qual_name)

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen35moe_gguf()


class ModelVariant(StrEnum):
    """Available Qwen 3.5 Creative GGUF model variants for causal language modeling."""

    QWEN_3_5_CREATIVE_26B_A3B_REAP_I1 = "26B_A3B_REAP_i1"
    QWEN_3_5_CREATIVE_26B_A3B_REAP = "26B_A3B_REAP"


class ModelLoader(ForgeModel):
    """Qwen 3.5 Creative GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-Creative-26B-A3B-REAP-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP: LLMModelConfig(
            pretrained_model_name="mradermacher/Qwen3.5-Creative-26B-A3B-REAP-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1

    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP_I1: "Qwen3.5-Creative-26B-A3B-REAP.i1-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_CREATIVE_26B_A3B_REAP: "Qwen3.5-Creative-26B-A3B-REAP.Q4_K_M.gguf",
    }

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
            model="Qwen 3.5 Creative GGUF",
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
        tokenizer_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

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
        model_kwargs["gguf_file"] = self._GGUF_FILES[self._variant]

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self._GGUF_FILES[self._variant]
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

    def _get_text_config(self):
        """Get the text config, handling both nested (MoE) and flat config structures."""
        if hasattr(self.config, "text_config"):
            return self.config.text_config
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
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
            self._variant_config.pretrained_model_name,
            gguf_file=self._GGUF_FILES[self._variant],
        )
        return self.config
