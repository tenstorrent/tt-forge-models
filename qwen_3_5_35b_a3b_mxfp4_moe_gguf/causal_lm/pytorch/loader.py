# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 35B-A3B MXFP4 MoE GGUF model loader implementation for causal language modeling.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _register_qwen35moe_gguf_support():
    """Register qwen35moe GGUF architecture support in transformers.

    The Qwen 3.5 MoE GGUF files use architecture identifier 'qwen35moe', but
    transformers does not yet include this mapping. We register it by reusing
    the qwen3_moe config/tensor mappings and patching load_gguf_checkpoint to
    map the architecture string correctly.
    """
    import transformers.integrations.ggml as ggml
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen3_5_moe" in ggml.GGUF_CONFIG_MAPPING:
        return

    # Register config mapping (same structure as qwen3_moe)
    ggml.GGUF_CONFIG_MAPPING["qwen3_5_moe"] = ggml.GGUF_CONFIG_MAPPING[
        "qwen3_moe"
    ].copy()

    # Register tensor processor
    gguf_utils.TENSOR_PROCESSORS["qwen35moe"] = gguf_utils.TENSOR_PROCESSORS["qwen3moe"]

    # Refresh supported architectures list
    gguf_utils.GGUF_SUPPORTED_ARCHITECTURES = list(
        gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"].keys()
    )

    # Register config defaults (same as qwen3_moe)
    if hasattr(ggml, "GGUF_CONFIG_DEFAULTS_MAPPING"):
        ggml.GGUF_CONFIG_DEFAULTS_MAPPING[
            "qwen3_5_moe"
        ] = ggml.GGUF_CONFIG_DEFAULTS_MAPPING.get("qwen3_moe", {}).copy()

    # Register tokenizer converter
    if hasattr(ggml, "GGUF_TO_FAST_CONVERTERS"):
        qwen3_converter = ggml.GGUF_TO_FAST_CONVERTERS.get("qwen3_moe")
        if qwen3_converter:
            ggml.GGUF_TO_FAST_CONVERTERS["qwen3_5_moe"] = qwen3_converter

    # Patch load_gguf_checkpoint to handle qwen35moe architecture string
    original_load = gguf_utils.load_gguf_checkpoint

    def _patched_load(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        orig_read_field = gguf_utils.read_field

        def _read_field_with_qwen35(reader, field):
            result = orig_read_field(reader, field)
            if field == "general.architecture" and result and result[0] == "qwen35moe":
                return ["qwen3moe"]
            return result

        gguf_utils.read_field = _read_field_with_qwen35
        try:
            result = original_load(
                gguf_checkpoint_path,
                return_tensors=return_tensors,
                model_to_load=model_to_load,
            )
            # Fix model_type from qwen3_moe to qwen3_5_moe
            if "config" in result and result["config"].get("model_type") == "qwen3_moe":
                result["config"]["model_type"] = "qwen3_5_moe"
            return result
        finally:
            gguf_utils.read_field = orig_read_field

    # Patch all modules that imported load_gguf_checkpoint
    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if (
            mod is not None
            and hasattr(mod, "load_gguf_checkpoint")
            and getattr(mod, "load_gguf_checkpoint") is original_load
        ):
            setattr(mod, "load_gguf_checkpoint", _patched_load)


_register_qwen35moe_gguf_support()

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
    """Available Qwen 3.5 35B-A3B MXFP4 MoE GGUF model variants for causal language modeling."""

    QWEN_3_5_35B_A3B_MXFP4_MOE_GGUF = "35B_A3B_MXFP4_MOE_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 3.5 35B-A3B MXFP4 MoE GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_MXFP4_MOE_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_MXFP4_MOE_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-MXFP4_MOE_BF16.gguf"

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
            model="Qwen 3.5 35B-A3B MXFP4 MoE GGUF",
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

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
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

        messages = [{"role": "user", "content": self.sample_text}]
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
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def _get_text_config(self):
        """Get the text config, handling both nested (MoE) and flat config structures."""
        if hasattr(self.config, "text_config"):
            return self.config.text_config
        return self.config

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
