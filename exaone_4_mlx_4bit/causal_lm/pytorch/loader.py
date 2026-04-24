# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 MLX 4-bit model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available EXAONE 4.0 MLX 4-bit model variants for causal language modeling."""

    EXAONE_4_0_32B_MLX_4BIT = "4.0_32B_MLX_4bit"


class ModelLoader(ForgeModel):
    """EXAONE 4.0 MLX 4-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_4_0_32B_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/EXAONE-4.0-32B-MLX-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_4_0_32B_MLX_4BIT

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
            model="EXAONE 4.0 MLX 4-bit",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _load_patched_config(self, pretrained_model_name):
        """Load config, computing layer_types from string sliding_window_pattern if needed.

        transformers 5.x passes config_dict to cls(**config_dict) before applying kwargs,
        so we must patch the dict before construction rather than relying on kwarg overrides.
        """
        from transformers import PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name)
        pattern = config_dict.get("sliding_window_pattern")
        if isinstance(pattern, str) and not config_dict.get("layer_types"):
            n = config_dict.get("num_hidden_layers", 0)
            char_to_type = {"L": "sliding_attention", "G": "full_attention"}
            config_dict["layer_types"] = [
                char_to_type.get(pattern[i % len(pattern)], "full_attention")
                for i in range(n)
            ]
            config_dict["sliding_window_pattern"] = len(pattern)
            model_type = config_dict.get("model_type", "")
            if model_type in CONFIG_MAPPING:
                return CONFIG_MAPPING[model_type].from_dict(config_dict)
        return AutoConfig.from_pretrained(pretrained_model_name)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = self._load_patched_config(pretrained_model_name)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        if hasattr(config, "quantization_config"):
            del config.quantization_config
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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = self._load_patched_config(
            self._variant_config.pretrained_model_name
        )
        return self.config
