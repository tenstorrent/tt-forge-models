# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 72B Embiggened GGUF model loader implementation for causal language modeling.
"""
import os

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

# Tokenizer repo that has config.json + tokenizer files (same vocab as the GGUF).
# Used for TT_RANDOM_WEIGHTS mode to avoid the full 25 GB GGUF download.
_TOKENIZER_FALLBACK = "Qwen/Qwen3-72B"

# Architecture extracted from GGUF metadata (qwen3.* keys).
# The "Embiggened" variant has a wider hidden_size (8192) than stock Qwen3-72B (7168).
_GGUF_ARCH = dict(
    hidden_size=8192,
    intermediate_size=29568,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    max_position_embeddings=40960,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    vocab_size=152064,
    tie_word_embeddings=False,
)


class ModelVariant(StrEnum):
    """Available Qwen3 72B Embiggened GGUF model variants for causal language modeling."""

    QWEN3_72B_EMBIGGENED_GGUF = "72B_EMBIGGENED_GGUF"


class ModelLoader(ForgeModel):
    """Qwen3 72B Embiggened GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_72B_EMBIGGENED_GGUF: LLMModelConfig(
            pretrained_model_name="QuixiAI/Qwen3-72B-Embiggened-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_72B_EMBIGGENED_GGUF

    GGUF_FILE = "Qwen3-72B-Embiggened-Q4_K_M.gguf"

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
            model="Qwen3 72B Embiggened GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_random_config(self):
        """Build a Qwen3Config from hardcoded GGUF metadata — avoids downloading the 25 GB GGUF."""
        from transformers import Qwen3Config

        arch = dict(_GGUF_ARCH)
        if self.num_layers is not None:
            arch["num_hidden_layers"] = self.num_layers
        return Qwen3Config(**arch)

    def _load_tokenizer(self, dtype_override=None):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            # The GGUF repo has no standalone tokenizer files — use the base Qwen3-72B
            # tokenizer (same vocabulary) to avoid the full GGUF download.
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_FALLBACK)
        else:
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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = self._build_random_config()
            model = AutoModelForCausalLM.from_config(config)
            if dtype_override is not None and isinstance(dtype_override, torch.dtype):
                model = model.to(dtype_override)
            model.eval()
            self.config = model.config
            self.model = model
            return model

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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.config = self._build_random_config()
            return self.config

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
