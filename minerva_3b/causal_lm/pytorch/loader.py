# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sapienza NLP Minerva 3B base model loader implementation for causal language modeling.

The upstream repo ``sapienzanlp/Minerva-3B-base-v1.0`` is gated.  To avoid a
hard dependency on repo access we bundle the model configuration locally and
fall back to it when the gated download fails.
"""
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, MistralConfig
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

logger = logging.getLogger(__name__)

# Local copy of the Minerva-3B-base-v1.0 config so we can compile without
# access to the gated HuggingFace repo.
_MINERVA_3B_CONFIG = {
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 8960,
    "max_position_embeddings": 16384,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "sliding_window": 2048,
    "tie_word_embeddings": False,
    "use_cache": False,
    "vocab_size": 32768,
}


class ModelVariant(StrEnum):
    """Available Minerva 3B model variants for causal language modeling."""

    MINERVA_3B_BASE_V1_0 = "3B_base_v1.0"


class ModelLoader(ForgeModel):
    """Sapienza NLP Minerva 3B base model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINERVA_3B_BASE_V1_0: LLMModelConfig(
            pretrained_model_name="sapienzanlp/Minerva-3B-base-v1.0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINERVA_3B_BASE_V1_0

    sample_text = (
        "What are the key differences between classical and quantum computing?"
    )

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
            model="Minerva 3B",
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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, **tokenizer_kwargs
            )
        except OSError:
            logger.warning(
                "Cannot access gated repo %s, falling back to mistralai/Mistral-7B-v0.1 tokenizer",
                self._variant_config.pretrained_model_name,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1", **tokenizer_kwargs
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _get_local_config(self):
        """Return a MistralConfig built from the bundled config dict."""
        return MistralConfig(**_MINERVA_3B_CONFIG)

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = self._get_local_config()
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            ).eval()
        except OSError:
            logger.warning(
                "Cannot access gated repo %s, instantiating from local config",
                pretrained_model_name,
            )
            config = self._get_local_config()
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            if dtype_override is not None:
                config.torch_dtype = dtype_override
            model = AutoModelForCausalLM.from_config(config).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        try:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        except OSError:
            logger.warning(
                "Cannot access gated repo %s, using local config",
                self._variant_config.pretrained_model_name,
            )
            self.config = self._get_local_config()
        return self.config
