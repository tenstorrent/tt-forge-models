# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TUM-EDA Flui3d-Chat Llama 3.3 Reasoning causal LM model loader implementation.

This is a LoRA adapter on meta-llama/Llama-3.3-70B-Instruct from
TUM-EDA/Flui3d-Chat-Llama3.3-Reasoning, fine-tuned for microfluidic chip
design generation with Chain-of-Thought reasoning. The adapter weights are
stored in the ``lora_model/`` subfolder of the repository.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
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
    """Available TUM-EDA Flui3d-Chat Llama 3.3 Reasoning variants for causal language modeling."""

    FLUI3D_CHAT_LLAMA3_3_REASONING = "Flui3d-Chat-Llama3.3-Reasoning"


class ModelLoader(ForgeModel):
    """TUM-EDA Flui3d-Chat Llama 3.3 Reasoning model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.FLUI3D_CHAT_LLAMA3_3_REASONING: LLMModelConfig(
            pretrained_model_name="TUM-EDA/Flui3d-Chat-Llama3.3-Reasoning",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUI3D_CHAT_LLAMA3_3_REASONING

    BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

    ADAPTER_SUBFOLDER = "lora_model"

    sample_text = (
        "Design a microfluidic chip with two inlets, one mixer, and a single outlet."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Flui3d-Chat-Llama3.3-Reasoning",
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
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(
            base_model, adapter_name, subfolder=self.ADAPTER_SUBFOLDER
        )
        model = model.merge_and_unload()
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

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
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)

        return self.config
