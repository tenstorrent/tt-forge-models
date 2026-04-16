# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimalism (salakash/Minimalism) LoRA model loader implementation for causal language modeling.

Note: This model uses MLX-format LoRA adapters (not PEFT), so we manually
download the adapter weights and merge them into the base model.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Minimalism model variants."""

    MINIMALISM = "Minimalism"


class ModelLoader(ForgeModel):
    """Minimalism LoRA model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINIMALISM: ModelConfig(
            pretrained_model_name="salakash/Minimalism",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINIMALISM

    BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Minimalism",
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
        return self.tokenizer

    @staticmethod
    def _merge_mlx_lora_adapters(model, adapter_repo):
        """Download MLX-format LoRA adapters and merge them into the base model."""
        config_path = hf_hub_download(adapter_repo, "adapter_config.json")
        with open(config_path) as f:
            adapter_config = json.load(f)
        scale = adapter_config["lora_parameters"]["scale"]

        adapter_path = hf_hub_download(adapter_repo, "adapters.safetensors")
        adapter_weights = {}
        with safe_open(adapter_path, framework="pt") as f:
            for key in f.keys():
                adapter_weights[key] = f.get_tensor(key)

        # Group lora_a and lora_b pairs by module path
        lora_pairs = {}
        for key in adapter_weights:
            if key.endswith(".lora_a"):
                module_path = key[: -len(".lora_a")]
                lora_pairs.setdefault(module_path, {})["lora_a"] = adapter_weights[key]
            elif key.endswith(".lora_b"):
                module_path = key[: -len(".lora_b")]
                lora_pairs.setdefault(module_path, {})["lora_b"] = adapter_weights[key]

        # MLX LoRA: output = x @ W^T + scale * x @ A @ B
        # Merge: W_merged = W + scale * B^T @ A^T
        state_dict = model.state_dict()
        for module_path, lora in lora_pairs.items():
            weight_key = module_path + ".weight"
            lora_a = lora["lora_a"].to(state_dict[weight_key].dtype)
            lora_b = lora["lora_b"].to(state_dict[weight_key].dtype)
            state_dict[weight_key] += scale * lora_b.T @ lora_a.T

        model.load_state_dict(state_dict)

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_repo = self._variant_config.pretrained_model_name
        self._merge_mlx_lora_adapters(model, adapter_repo)

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [
            {"role": "user", "content": "write a quick sort algorithm."},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])
