# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit model loader implementation for causal language modeling.
"""
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


def _dequantize_mlx_4bit(w_q, scales, biases, group_size=64, bits=4):
    """Dequantize MLX 4-bit packed weights to bfloat16.

    MLX stores quantized weights as packed uint32 tensors with per-group scales and biases.
    Dequantization formula: w = q * scale + bias, where q is an unsigned 4-bit integer (0-15).
    """
    out_dim, in_dim_packed = w_q.shape
    in_dim = in_dim_packed * (32 // bits)

    shift = torch.arange(0, 32, bits, dtype=torch.int32, device=w_q.device)
    w_unpacked = (w_q.unsqueeze(-1).to(torch.int32) >> shift) & ((1 << bits) - 1)
    w_unpacked = w_unpacked.reshape(out_dim, in_dim).to(torch.float32)

    scales_f = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    biases_f = biases.to(torch.float32).repeat_interleave(group_size, dim=1)

    return (w_unpacked * scales_f + biases_f).to(torch.bfloat16)


def _load_mlx_state_dict(pretrained_model_name, dtype=torch.bfloat16):
    """Load and dequantize an MLX 4-bit safetensors file into a standard state dict."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(pretrained_model_name, "model.safetensors.index.json")
    import json

    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))

    raw = {}
    for shard in shard_files:
        shard_path = hf_hub_download(pretrained_model_name, shard)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    state_dict = {}
    quantized_bases = set()
    for key in raw:
        if key.endswith(".scales") or key.endswith(".biases"):
            base = key.rsplit(".", 1)[0]
            quantized_bases.add(base)

    for base in quantized_bases:
        w_q = raw[base + ".weight"]
        scales = raw[base + ".scales"]
        biases = raw[base + ".biases"]

        config_key = base + ".quantization_config"
        group_size = 64
        bits = 4

        state_dict[base + ".weight"] = _dequantize_mlx_4bit(
            w_q, scales, biases, group_size=group_size, bits=bits
        )

    for key, tensor in raw.items():
        base = key.rsplit(".", 1)[0]
        suffix = key.rsplit(".", 1)[1]
        if suffix in ("scales", "biases"):
            continue
        if base in quantized_bases:
            continue
        state_dict[key] = tensor.to(dtype) if tensor.is_floating_point() else tensor

    return state_dict


class ModelVariant(StrEnum):
    """Available mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit model variants for causal language modeling."""

    DEEPSEEK_R1_0528_QWEN3_8B_4BIT = "DeepSeek_R1_0528_Qwen3_8B_4bit"


class ModelLoader(ForgeModel):
    """mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_R1_0528_QWEN3_8B_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_R1_0528_QWEN3_8B_4BIT

    sample_text = "Please reason step by step. What is 25 multiplied by 16?"

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
            model="DeepSeek-R1-0528-Qwen3-8B-4bit MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.quantization_config = None

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

        state_dict = _load_mlx_state_dict(pretrained_model_name, dtype=dtype)
        model.load_state_dict(state_dict, strict=True)
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
