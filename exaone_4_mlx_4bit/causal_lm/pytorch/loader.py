# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 MLX 4-bit model loader implementation for causal language modeling.
"""
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
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

    def _load_exaone4_config(self, pretrained_model_name: str) -> Exaone4Config:
        """Load Exaone4Config, working around the transformers 5.2 bug where
        string sliding_window_pattern causes TypeError in the config constructor."""
        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        pattern = config_dict.get("sliding_window_pattern")
        if isinstance(pattern, str) and config_dict.get("layer_types") is None:
            mapping = {"L": "sliding_attention", "G": "full_attention"}
            num_layers = config_dict["num_hidden_layers"]
            config_dict["layer_types"] = [
                mapping[pattern[i % len(pattern)]] for i in range(num_layers)
            ]
        config_dict.pop("transformers_version", None)
        return Exaone4Config(**config_dict)

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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = self._load_exaone4_config(pretrained_model_name)
        if self.num_layers is not None:
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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.config = self._load_exaone4_config(pretrained_model_name)
        return self.config
