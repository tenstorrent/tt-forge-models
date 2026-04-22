# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
L3.3 The Omega Directive 70B GGUF model loader implementation for causal language modeling.
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

# Llama 3.3 70B architecture parameters used when TT_RANDOM_WEIGHTS skips GGUF loading
_LLAMA_70B_HIDDEN_SIZE = 8192
_LLAMA_70B_INTERMEDIATE_SIZE = 28672
_LLAMA_70B_NUM_LAYERS = 80
_LLAMA_70B_NUM_HEADS = 64
_LLAMA_70B_NUM_KV_HEADS = 8
_LLAMA_70B_MAX_POS = 131072
_LLAMA_70B_VOCAB_SIZE = 128256
_LLAMA_70B_ROPE_THETA = 500000.0


class ModelVariant(StrEnum):
    """Available L3.3 The Omega Directive 70B GGUF model variants for causal language modeling."""

    L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M = "GGUF_Q4_K_M"
    L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF = (
        "Unslop_v2_1_heretic_GGUF_Q4_K_M"
    )


class ModelLoader(ForgeModel):
    """L3.3 The Omega Directive 70B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/L3.3-The-Omega-Directive-70B-Unslop-v2.0-heretic-i1-GGUF",
            max_length=128,
        ),
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/L3.3-The-Omega-Directive-70B-Unslop-v2.1-heretic-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M

    _GGUF_FILES = {
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_GGUF_Q4_K_M: "L3.3-The-Omega-Directive-70B-Unslop-v2.0-heretic.i1-Q4_K_M.gguf",
        ModelVariant.L33_THE_OMEGA_DIRECTIVE_70B_UNSLOP_V2_1_HERETIC_GGUF: "L3.3-The-Omega-Directive-70B-Unslop-v2.1-heretic.Q4_K_M.gguf",
    }

    sample_text = "What is your favorite city?"

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
            model="L3.3 The Omega Directive 70B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    def _make_llama_config(self):
        from transformers import LlamaConfig

        return LlamaConfig(
            hidden_size=_LLAMA_70B_HIDDEN_SIZE,
            intermediate_size=_LLAMA_70B_INTERMEDIATE_SIZE,
            num_hidden_layers=self.num_layers or _LLAMA_70B_NUM_LAYERS,
            num_attention_heads=_LLAMA_70B_NUM_HEADS,
            num_key_value_heads=_LLAMA_70B_NUM_KV_HEADS,
            max_position_embeddings=_LLAMA_70B_MAX_POS,
            vocab_size=_LLAMA_70B_VOCAB_SIZE,
            rope_theta=_LLAMA_70B_ROPE_THETA,
        )

    def _load_tokenizer(self, dtype_override=None):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            # Avoid reading the 42.5 GB GGUF file just for tokenizer metadata;
            # load_inputs will synthesise fake token ids instead.
            return None

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self._gguf_file

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            from transformers import LlamaForCausalLM

            config = self._make_llama_config()
            dtype = dtype_override or torch.bfloat16
            old_default = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                model = LlamaForCausalLM(config)
            finally:
                torch.set_default_dtype(old_default)
        else:
            if self.tokenizer is None:
                self._load_tokenizer(dtype_override=dtype_override)

            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self._gguf_file

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self._gguf_file
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
        max_length = self._variant_config.max_length

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            # Synthesise token ids to avoid loading the 42.5 GB GGUF file.
            input_ids = torch.randint(
                0, _LLAMA_70B_VOCAB_SIZE, (batch_size, max_length)
            )
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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

    def load_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.config = self._make_llama_config()
            return self.config

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self._gguf_file
        )
        return self.config

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
