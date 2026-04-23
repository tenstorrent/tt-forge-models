# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/translategemma-4b-it-4bit model loader implementation for text translation.

A 4-bit MLX-quantized variant of google/translategemma-4b-it, a Gemma3
conditional generation model fine-tuned for multilingual translation.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
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
    """Available mlx-community/translategemma-4b-it-4bit model variants."""

    TRANSLATEGEMMA_4B_IT_4BIT = "translategemma-4b-it-4bit"


def _mlx_dequantize(weight_uint32, scales, biases, group_size=64, bits=4):
    """Dequantize MLX 4-bit affine quantized weight tensor to bfloat16."""
    M, N_packed = weight_uint32.shape
    values_per_word = 32 // bits  # 8 for 4-bit
    N = N_packed * values_per_word

    shifts = torch.arange(values_per_word, dtype=torch.int32)
    unpacked = (weight_uint32.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF
    weight_float = unpacked.reshape(M, N).to(scales.dtype)

    num_groups = N // group_size
    weight_grouped = weight_float.view(M, num_groups, group_size)
    dequant = weight_grouped * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    return dequant.view(M, N)


def _load_mlx_state_dict(model_name, group_size=64, bits=4):
    """Load and dequantize MLX quantized safetensors into a float state dict."""
    idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = list(set(weight_map.values()))
    raw_tensors = {}
    for shard in shard_files:
        shard_path = hf_hub_download(model_name, shard)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                raw_tensors[key] = f.get_tensor(key)

    state_dict = {}
    processed = set()

    for key in list(raw_tensors):
        if key in processed:
            continue
        if key.endswith(".scales") or key.endswith(".biases"):
            processed.add(key)
            continue

        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            scales_key = base + ".scales"
            biases_key = base + ".biases"
            if scales_key in raw_tensors and biases_key in raw_tensors:
                w = raw_tensors[key]
                if w.dtype == torch.uint32:
                    dequantized = _mlx_dequantize(
                        w,
                        raw_tensors[scales_key],
                        raw_tensors[biases_key],
                        group_size=group_size,
                        bits=bits,
                    )
                    state_dict[key] = dequantized
                    processed.update({key, scales_key, biases_key})
                    continue

        tensor = raw_tensors[key]
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        state_dict[key] = tensor
        processed.add(key)

    return state_dict


class ModelLoader(ForgeModel):
    """mlx-community/translategemma-4b-it-4bit model loader for text translation."""

    _VARIANTS = {
        ModelVariant.TRANSLATEGEMMA_4B_IT_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/translategemma-4b-it-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSLATEGEMMA_4B_IT_4BIT

    sample_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "cs",
                    "target_lang_code": "de-DE",
                    "text": "V nejhorším případě i k prasknutí čočky.",
                }
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mlx-community translategemma-4b-it-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override or torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        # Build model from config, stripping MLX quantization_config so that
        # standard PyTorch can construct the architecture.
        cfg = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(cfg, "quantization_config"):
            del cfg.quantization_config

        model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)

        # Load and dequantize the MLX 4-bit weights, then inject into model.
        state_dict = _load_mlx_state_dict(pretrained_model_name)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer.apply_chat_template(
            self.sample_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
