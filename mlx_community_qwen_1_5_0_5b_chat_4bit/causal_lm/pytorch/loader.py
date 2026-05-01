# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen1.5-0.5B-Chat-4bit model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoTokenizer, AutoConfig, Qwen2ForCausalLM
from typing import Optional

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


def _mlx_dequantize(weight, scales, biases):
    """Dequantize MLX-format packed-integer weights to bfloat16.

    MLX affine quantization packs (32//bits) unsigned int values per uint32
    element, LSB first. Bit-width is inferred from tensor shapes:
        n_per_elem = in_dim / in_packed
              bits = 32 / n_per_elem
    Each group of group_size original values shares one scale and bias:
        original = packed_int * scale + bias
    """
    out_dim, in_packed = weight.shape
    group_size = 64  # from config quantization.group_size
    in_dim = scales.shape[1] * group_size
    n_per_elem = in_dim // in_packed  # 8 for 4-bit
    bits = 32 // n_per_elem

    weight_i32 = weight.view(torch.int32)
    mask = (1 << bits) - 1

    unpacked = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for shift in range(n_per_elem):
        unpacked[:, shift::n_per_elem] = ((weight_i32 >> (shift * bits)) & mask).float()

    scales_f = scales.float().repeat_interleave(group_size, dim=1)
    biases_f = biases.float().repeat_interleave(group_size, dim=1)
    return (unpacked * scales_f + biases_f).to(torch.bfloat16)


def _load_mlx_state_dict(pretrained_model_name, dtype):
    """Load the MLX safetensors checkpoint and dequantize packed-int weights."""
    from huggingface_hub import snapshot_download
    import os
    from safetensors import safe_open

    local_dir = snapshot_download(pretrained_model_name)

    sf_files = sorted(
        [
            os.path.join(local_dir, f)
            for f in os.listdir(local_dir)
            if f.endswith(".safetensors")
        ]
    )
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {local_dir}")

    # Collect all keys to identify quantized weights
    all_keys = []
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt") as f:
            all_keys.extend(f.keys())

    quantized_bases = {
        k[: -len(".weight")]
        for k in all_keys
        if k.endswith(".weight") and (k[: -len(".weight")] + ".scales") in all_keys
    }

    # Build shard lookup
    key_to_shard = {}
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt") as f:
            for k in f.keys():
                key_to_shard[k] = sf_path

    open_shards = {p: safe_open(p, framework="pt") for p in sf_files}

    def _get_tensor(key):
        return open_shards[key_to_shard[key]].get_tensor(key)

    state_dict = {}
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(".scales") or key.endswith(".biases"):
                    continue

                tensor = f.get_tensor(key)
                base = key[: -len(".weight")] if key.endswith(".weight") else None

                if base is not None and base in quantized_bases:
                    scales = _get_tensor(base + ".scales")
                    biases = _get_tensor(base + ".biases")
                    tensor = _mlx_dequantize(tensor, scales, biases)
                else:
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype)

                state_dict[key] = tensor

    for f in open_shards.values():
        del f

    return state_dict


class ModelVariant(StrEnum):
    """Available mlx-community/Qwen1.5-0.5B-Chat-4bit model variants for causal LM."""

    QWEN_1_5_0_5B_CHAT_4BIT = "0_5B_Chat_4bit"


class ModelLoader(ForgeModel):
    """mlx-community/Qwen1.5-0.5B-Chat-4bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_1_5_0_5B_CHAT_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen1.5-0.5B-Chat-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_1_5_0_5B_CHAT_4BIT

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="mlx-community-Qwen1.5-0.5B-Chat-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config and strip MLX quantization key (not a transformers quantizer)
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization"):
            del config.quantization

        # Build empty model with correct architecture
        model = Qwen2ForCausalLM(config)
        model = model.to(dtype)

        # Load and dequantize MLX 4-bit safetensors
        state_dict = _load_mlx_state_dict(pretrained_model_name, dtype)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if hasattr(inputs[key], "repeat_interleave"):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
