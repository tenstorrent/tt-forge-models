# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5 FP8 model loader implementation for causal language modeling.
"""
import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _load_fp8_weights(model_dir: str, target_dtype: torch.dtype) -> dict:
    """Load safetensors shards, converting float8 tensors to target_dtype."""
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".safetensors")
        )

    state_dict = {}
    for fname in shard_files:
        shard = load_file(os.path.join(model_dir, fname), device="cpu")
        for key, tensor in shard.items():
            if tensor.is_floating_point() and tensor.dtype not in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
            ):
                tensor = tensor.to(target_dtype)
            state_dict[key] = tensor
    return state_dict


class ModelVariant(StrEnum):
    """Available GLM-5 FP8 model variants for causal language modeling."""

    GLM_5_FP8 = "FP8"


class ModelLoader(ForgeModel):
    """GLM-5 FP8 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_5_FP8: LLMModelConfig(
            pretrained_model_name="unsloth/GLM-5-FP8",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_5_FP8

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
            model="GLM-5 FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        # Load config and strip FP8 quantization to bypass triton dependency on CPU
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        config.quantization_config = None

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        # Create model from config with no quantization (random weights)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(target_dtype)

        # Download (or find cached) model files and load weights, converting float8 to target_dtype
        model_dir = snapshot_download(pretrained_model_name)
        state_dict = _load_fp8_weights(model_dir, target_dtype)
        model.load_state_dict(state_dict, strict=False)

        model = model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
