# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
alexdenton Qwen3.5 35B A3B Heretic GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


import transformers.modeling_gguf_pytorch_utils as _gguf_utils


def _patch_transformers_qwen35moe():
    """Register qwen35moe GGUF architecture (Qwen3.5 MoE) with transformers."""
    from transformers.integrations.ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_CONFIG_DEFAULTS_MAPPING,
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        TENSOR_PROCESSORS,
        Qwen2MoeTensorProcessor,
    )
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )

    import numpy as _np

    class Qwen35MoeTensorProcessor(Qwen2MoeTensorProcessor):
        """Extends Qwen2MoeTensorProcessor with conv1d weight reshape for Qwen3.5 MoE SSM layers."""

        def process(self, weights, name: str, **kwargs):
            # GGUF stores ssm_conv1d.weight as [N, kernel] but HF expects [N, 1, kernel]
            if "ssm_conv1d" in name and weights.ndim == 2:
                weights = weights[:, _np.newaxis, :]
            return super().process(weights, name, **kwargs)

    arch = "qwen35moe"
    if arch in GGUF_SUPPORTED_ARCHITECTURES:
        return

    qwen35moe_config_mapping = GGUF_CONFIG_MAPPING["qwen3_moe"].copy()
    qwen35moe_config_mapping["full_attention_interval"] = "full_attention_interval"
    GGUF_CONFIG_MAPPING[arch] = qwen35moe_config_mapping
    GGUF_CONFIG_DEFAULTS_MAPPING[arch] = GGUF_CONFIG_DEFAULTS_MAPPING.get(
        "qwen3_moe", {}
    ).copy()
    GGUF_TO_FAST_CONVERTERS[arch] = GGUFQwen2Converter
    GGUF_SUPPORTED_ARCHITECTURES.append(arch)
    TENSOR_PROCESSORS[arch] = Qwen35MoeTensorProcessor
    CONFIG_MAPPING.register(arch, Qwen3_5MoeConfig, exist_ok=True)

    import re as _re

    _orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map
    _fused_gate_up_pattern = _re.compile(r"(blk\.\d+\.ffn_)gate_up(_exps.*)")

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
            model_type = arch
        result = _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )
        # The GGUF file stores expert projections as separate gate/up tensors
        # (ffn_gate_exps, ffn_up_exps), but the qwen35moe tensor name map assumes
        # the fused format (ffn_gate_up_exps). Add the separate-format mappings so
        # they can be loaded and merged into the fused HF parameter.
        additions = {}
        for key, hf_name in result.items():
            m = _fused_gate_up_pattern.fullmatch(key)
            if m:
                additions[m.group(1) + "gate" + m.group(2)] = hf_name
                additions[m.group(1) + "up" + m.group(2)] = hf_name
        result.update(additions)
        return result

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_transformers_qwen35moe()

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
    """Available alexdenton Qwen3.5 35B A3B Heretic GGUF model variants for causal language modeling."""

    QWEN_3_5_35B_A3B_HERETIC_GGUF = "35B_A3B_HERETIC_GGUF"


class ModelLoader(ForgeModel):
    """alexdenton Qwen3.5 35B A3B Heretic GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="alexdenton/Qwen3.5-35B-A3B-heretic-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_HERETIC_GGUF

    GGUF_FILE = "Qwen3.5-35B-A3B-heretic-Q4_K_M.gguf"

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
            model="alexdenton Qwen3.5 35B A3B Heretic GGUF",
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
