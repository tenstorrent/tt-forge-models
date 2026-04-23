# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 32B Instruct GGUF model loader implementation for causal language modeling.
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
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
    """Available Qwen 2.5 32B Instruct GGUF model variants for causal language modeling."""

    QWEN_2_5_32B_INSTRUCT_GGUF = "32B_Instruct_GGUF"
    THERAINS_QWEN_2_5_32B_INSTRUCT_Q4_K_M_GGUF = "TheRains_32B_Instruct_Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Qwen 2.5 32B Instruct GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_32B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen2.5-32B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.THERAINS_QWEN_2_5_32B_INSTRUCT_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="TheRains/Qwen2.5-32B-Instruct-Q4_K_M-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_32B_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_2_5_32B_INSTRUCT_GGUF: "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        ModelVariant.THERAINS_QWEN_2_5_32B_INSTRUCT_Q4_K_M_GGUF: "qwen2.5-32b-instruct-q4_k_m.gguf",
    }

    # Non-GGUF model for config/tokenizer when TT_RANDOM_WEIGHTS is set
    _RANDOM_WEIGHTS_PRETRAINED_NAME = "Qwen/Qwen2.5-32B-Instruct"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers
        self.gguf_file = self._GGUF_FILES[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen 2.5 32B Instruct GGUF",
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

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._RANDOM_WEIGHTS_PRETRAINED_NAME, **tokenizer_kwargs
            )
        else:
            tokenizer_kwargs["gguf_file"] = self.gguf_file
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, **tokenizer_kwargs
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(self._RANDOM_WEIGHTS_PRETRAINED_NAME)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model = Qwen2ForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            pretrained_model_name = self._variant_config.pretrained_model_name
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["gguf_file"] = self.gguf_file

            if self.num_layers is not None:
                config = AutoConfig.from_pretrained(
                    pretrained_model_name, gguf_file=self.gguf_file
                )
                config.num_hidden_layers = self.num_layers
                model_kwargs["config"] = config

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
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
            self.config = AutoConfig.from_pretrained(
                self._RANDOM_WEIGHTS_PRETRAINED_NAME
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name, gguf_file=self.gguf_file
            )
        return self.config
