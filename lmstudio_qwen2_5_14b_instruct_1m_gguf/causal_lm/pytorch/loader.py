# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF model loader implementation for causal language modeling.
"""
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
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
    """Available lmstudio Qwen2.5 14B Instruct 1M GGUF model variants for causal language modeling."""

    QWEN2_5_14B_INSTRUCT_1M_GGUF = "14B_Instruct_1M_GGUF"


class ModelLoader(ForgeModel):
    """lmstudio-community Qwen2.5 14B Instruct 1M GGUF model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN2_5_14B_INSTRUCT_1M_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN2_5_14B_INSTRUCT_1M_GGUF

    GGUF_FILE = "Qwen2.5-14B-Instruct-1M-Q4_K_M.gguf"

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
            model="lmstudio Qwen2.5 14B Instruct 1M GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _qwen2_5_14b_config(self):
        return Qwen2Config(
            vocab_size=152064,
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=self.num_layers if self.num_layers is not None else 48,
            num_attention_heads=40,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=False,
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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = self._qwen2_5_14b_config()
            target_dtype = (
                dtype_override if dtype_override is not None else torch.bfloat16
            )
            orig_dtype = torch.get_default_dtype()
            torch.set_default_dtype(target_dtype)
            try:
                model = Qwen2ForCausalLM(config)
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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            vocab_size = 152064
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
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.config = self._qwen2_5_14b_config()
            return self.config
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
