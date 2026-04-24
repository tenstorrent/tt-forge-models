# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
bartowski c4ai-command-r-plus GGUF model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import CohereConfig, CohereForCausalLM
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
    """Available bartowski c4ai-command-r-plus GGUF model variants for causal language modeling."""

    C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF = "C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """bartowski c4ai-command-r-plus GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/c4ai-command-r-plus-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.C4AI_COMMAND_R_PLUS_Q4_K_M_GGUF

    GGUF_FILE = "c4ai-command-r-plus-Q4_K_M.gguf"

    sample_text = "What is the capital of France?"

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
            model="bartowski c4ai-command-r-plus GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _use_random_weights():
        return os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        )

    def _c4ai_command_r_plus_config(self):
        return CohereConfig(
            vocab_size=256000,
            hidden_size=8192,
            intermediate_size=22528,
            num_hidden_layers=self.num_layers if self.num_layers is not None else 64,
            num_attention_heads=64,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_theta=2.0e5,
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

        if self._use_random_weights():
            config = self._c4ai_command_r_plus_config()
            target_dtype = (
                dtype_override if dtype_override is not None else torch.bfloat16
            )
            orig_dtype = torch.get_default_dtype()
            torch.set_default_dtype(target_dtype)
            try:
                model = CohereForCausalLM(config)
            finally:
                torch.set_default_dtype(orig_dtype)
            model.eval()
            self.config = model.config
            self.model = model
            return model

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
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        max_length = self._variant_config.max_length

        if self._use_random_weights():
            vocab_size = 256000
            input_ids = torch.randint(0, vocab_size, (batch_size, max_length))
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
        if self._use_random_weights():
            self.config = self._c4ai_command_r_plus_config()
            return self.config
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
