# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
arthurcollet Qwen3.5 27B MLX NVFP4 model loader for causal language modeling.
"""

import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

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


def _dequantize_nvfp4(
    weight_uint32: torch.Tensor, scales_uint8: torch.Tensor
) -> torch.Tensor:
    """Dequantize nvfp4: unpack E2M1 fp4 from uint32, apply fp8 E4M3fn group scales.

    weight_uint32: (rows, cols_packed) uint32 — 8 fp4 nibbles per word.
    scales_uint8:  (rows, num_groups) uint8  — one fp8 E4M3fn scale per 16 fp4 values.
    Returns bfloat16 tensor of shape (rows, cols_packed * 8).
    """
    rows, cols_packed = weight_uint32.shape

    # Unpack fp4 nibbles: low nibble first, then high nibble of each byte.
    weight_bytes = weight_uint32.view(torch.uint8)  # (rows, cols_packed*4)
    low = (weight_bytes & 0xF).long()
    high = ((weight_bytes >> 4) & 0xF).long()
    nibbles = torch.stack([low, high], dim=-1).reshape(rows, -1)  # (rows, cols_packed*8)

    # E2M1 fp4: bit3=sign, bits[2:1]=exponent(bias=1), bit0=mantissa
    sign = ((nibbles >> 3) & 1).float()
    exp = (nibbles >> 1) & 3
    mant = (nibbles & 1).float()
    magnitude = torch.where(
        exp > 0,
        (2.0 ** (exp.float() - 1.0)) * (1.0 + 0.5 * mant),
        0.5 * mant,
    )
    fp4_vals = (1.0 - 2.0 * sign) * magnitude  # (rows, cols_packed*8)

    # fp8 E4M3fn group scales: sign(1b) exp(4b,bias=7) mantissa(3b)
    s_sign = ((scales_uint8 >> 7) & 1).float()
    s_exp = (scales_uint8 >> 3) & 0xF
    s_mant = (scales_uint8 & 0x7).float()
    s_mag = torch.where(
        s_exp > 0,
        (2.0 ** (s_exp.float() - 7.0)) * (1.0 + s_mant / 8.0),
        (2.0 ** (-6.0)) * (s_mant / 8.0),
    )
    scales_f = (1.0 - 2.0 * s_sign) * s_mag  # (rows, num_groups)

    # One scale per group_size=16 fp4 values.
    group_size = 16
    num_groups = nibbles.shape[1] // group_size
    dequant = (
        fp4_vals.reshape(rows, num_groups, group_size) * scales_f.unsqueeze(-1)
    ).reshape(rows, -1)
    return dequant.to(torch.bfloat16)


class ModelVariant(StrEnum):
    """Available arthurcollet Qwen3.5 27B MLX NVFP4 model variants."""

    QWEN3_5_27B_MLX_NVFP4 = "Qwen3_5_27B_mlx_nvfp4"


class ModelLoader(ForgeModel):
    """arthurcollet Qwen3.5 27B MLX NVFP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_27B_MLX_NVFP4: LLMModelConfig(
            pretrained_model_name="arthurcollet/Qwen3.5-27B-mlx-nvfp4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_27B_MLX_NVFP4

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
            model="arthurcollet Qwen3.5 27B MLX NVFP4",
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

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The checkpoint was saved with Qwen3_5ForCausalLM wrapped under .language_model
        # inside a VL container.  The top-level config is Qwen3_5Config (VL); the text
        # backbone config lives at config.text_config.
        full_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = full_config.text_config

        # Locate sharded safetensors files (downloads to HF cache if not present).
        model_dir = snapshot_download(pretrained_model_name)
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)

        # Read every tensor whose key starts with "language_model." from all shards.
        PREFIX = "language_model."
        SCALES_SUFFIX = ".scales"
        all_raw: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_index["weight_map"].values())):
            shard_path = os.path.join(model_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith(PREFIX):
                        all_raw[key] = sf.get_tensor(key)

        # Build state dict for Qwen3_5ForCausalLM:
        #   - strip "language_model." prefix
        #   - dequantize uint32 (packed nvfp4) weights using their uint8 (fp8 E4M3fn) scales
        #   - leave bfloat16 / float32 tensors as-is
        state_dict: dict[str, torch.Tensor] = {}
        for key, tensor in all_raw.items():
            if key.endswith(SCALES_SUFFIX):
                continue  # processed together with the weight tensor below
            model_key = key[len(PREFIX):]
            if tensor.dtype == torch.uint32:
                scales_key = key.rsplit(".", 1)[0] + SCALES_SUFFIX
                scales = all_raw[scales_key]
                tensor = _dequantize_nvfp4(tensor, scales)
            state_dict[model_key] = tensor

        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(text_config)
        model = model.to_empty(device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        model = model.eval()
        if missing:
            raise RuntimeError(
                f"load_state_dict: {len(missing)} missing keys after nvfp4 load; "
                f"first 5: {missing[:5]}"
            )

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
            if layer.layer_type == "full_attention":
                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            elif layer.layer_type == "linear_attention":
                shard_specs[layer.linear_attn.in_proj_qkv.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_z.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_b.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.in_proj_a.weight] = ("model", "batch")
                shard_specs[layer.linear_attn.out_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
