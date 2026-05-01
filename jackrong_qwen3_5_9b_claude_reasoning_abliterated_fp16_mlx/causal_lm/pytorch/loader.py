# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jackrong Qwen3.5-9B Claude Reasoning Abliterated fp16 MLX model loader implementation for causal language modeling.
"""

import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers import Qwen3_5ForCausalLM
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
    """Available Jackrong Qwen3.5-9B Claude Reasoning Abliterated fp16 MLX model variants for causal language modeling."""

    JACKRONG_QWEN3_5_9B_CLAUDE_REASONING_ABLITERATED_FP16_MLX = (
        "9B_Claude_Reasoning_Abliterated_fp16_MLX"
    )


class ModelLoader(ForgeModel):
    """Jackrong Qwen3.5-9B Claude Reasoning Abliterated fp16 MLX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.JACKRONG_QWEN3_5_9B_CLAUDE_REASONING_ABLITERATED_FP16_MLX: LLMModelConfig(
            pretrained_model_name="AITRADER/Jackrong-Qwen3.5-9B-Claude-Reasoning-abliterated-fp16-MLX",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.JACKRONG_QWEN3_5_9B_CLAUDE_REASONING_ABLITERATED_FP16_MLX
    )

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
            model="Jackrong Qwen3.5-9B Claude Reasoning Abliterated fp16 MLX",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The checkpoint stores weights in VLM layout (language_model.* prefix)
        # because it was saved as Qwen3_5ForConditionalGeneration.  AutoModelForCausalLM
        # resolves to Qwen3_5ForCausalLM (text-only, expects flat model.* / lm_head.*
        # keys), so the keys never match and the model ends up with random weights.
        # Fix: download the safetensors shards, strip the "language_model." prefix,
        # and load the remapped state dict into Qwen3_5ForCausalLM directly.
        full_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = full_config.text_config

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers

        model_dir = snapshot_download(pretrained_model_name)
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        state_dict = {}
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_sd = load_file(os.path.join(model_dir, shard_file))
            for k, v in shard_sd.items():
                if k.startswith("language_model."):
                    state_dict[k[len("language_model."):]] = v

        torch_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = Qwen3_5ForCausalLM(text_config)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(torch_dtype).eval()

        self.config = text_config
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

            if hasattr(layer, "self_attn"):
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif hasattr(layer, "linear_attn"):
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
