# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
H2O Danube3 model loader implementation for causal language modeling.
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
    """Available H2O Danube3 model variants for causal language modeling."""

    H2O_DANUBE3_4B_CHAT = "4B_Chat"
    H2O_DANUBE3_4B_CHAT_Q4_K_M = "4B_Chat_Q4_K_M"
    H2O_DANUBE3_4B_BASE = "4B_Base"


class ModelLoader(ForgeModel):
    """H2O Danube3 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.H2O_DANUBE3_4B_CHAT: LLMModelConfig(
            pretrained_model_name="h2oai/h2o-danube3-4b-chat",
            max_length=128,
        ),
        ModelVariant.H2O_DANUBE3_4B_CHAT_Q4_K_M: LLMModelConfig(
            pretrained_model_name="h2oai/h2o-danube3-4b-chat-GGUF",
            max_length=128,
        ),
        ModelVariant.H2O_DANUBE3_4B_BASE: LLMModelConfig(
            pretrained_model_name="h2oai/h2o-danube3-4b-base",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.H2O_DANUBE3_4B_CHAT_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.H2O_DANUBE3_4B_CHAT_Q4_K_M: "h2o-danube3-4b-chat-Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    def _is_gguf_variant(self):
        return self._variant in self._GGUF_FILES

    @property
    def gguf_file(self):
        return self._GGUF_FILES.get(self._variant)

    def _is_gguf_variant(self):
        return self._variant in self._GGUF_FILES

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant in cls._GGUF_FILES:
            model_name = "H2O Danube3 GGUF"
        else:
            model_name = "H2O Danube3"
        return ModelInfo(
            model="H2O Danube3",
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
        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self.gguf_file

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
        if self._is_gguf_variant():
            model_kwargs["gguf_file"] = self.gguf_file

        if self.num_layers is not None:
            config_kwargs = {}
            if self._is_gguf_variant():
                config_kwargs["gguf_file"] = self.gguf_file
            config = AutoConfig.from_pretrained(pretrained_model_name, **config_kwargs)
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

        inputs = self.tokenizer(
            [self.sample_text],
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
        config_kwargs = {}
        if self._is_gguf_variant():
            config_kwargs["gguf_file"] = self.gguf_file
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, **config_kwargs
        )
        return self.config
