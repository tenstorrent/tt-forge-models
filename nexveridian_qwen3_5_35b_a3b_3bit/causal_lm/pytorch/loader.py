# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NexVeridian/Qwen3.5-35B-A3B-3bit MLX model loader for causal language modeling.
"""
import json
import os
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForCausalLM

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


def _dequantize_mlx_affine(
    weight_uint32: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    bits: int,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize MLX affine group quantization for any bit width.

    weight_uint32: (rows, cols_packed) uint32 — bits-per-value, packed LSB-first.
    scales:  (rows, num_groups) bfloat16 — scale per group.
    biases:  (rows, num_groups) bfloat16 — bias (zero-point) per group.
    bits:    number of bits per weight value (3 or 8).
    Returns bfloat16 tensor of shape (rows, in_features) where in_features is
    determined from cols_packed and bits.
    """
    rows, cols_packed = weight_uint32.shape
    mask = (1 << bits) - 1

    if 32 % bits == 0:
        # Exact packing: bits divides 32, so no cross-boundary values.
        # View as the smallest integer type that holds `bits` bits.
        in_features = cols_packed * (32 // bits)
        w_bytes = weight_uint32.cpu().numpy().view(np.uint8)  # (rows, cols_packed*4)
        if bits == 8:
            # Each byte is one 8-bit weight.
            main = w_bytes.astype(np.uint32)  # (rows, in_features)
        else:
            # bits == 4: two nibbles per byte.
            low = (w_bytes & 0xF).astype(np.uint32)
            high = ((w_bytes >> 4) & 0xF).astype(np.uint32)
            main = np.stack([low, high], axis=-1).reshape(rows, -1)
    else:
        # Non-exact packing (e.g. bits==3): values span uint32 boundaries.
        in_features = (cols_packed * 32) // bits
        w_np = weight_uint32.cpu().numpy().view(np.uint32)
        bit_positions = np.arange(in_features, dtype=np.int64) * bits
        w32_idx = (bit_positions // 32).astype(np.int32)
        bit_off = (bit_positions % 32).astype(np.int32)
        main = (w_np[:, w32_idx] >> bit_off) & mask
        # Fix values that span a uint32 boundary.
        for i in np.where(bit_off + bits > 32)[0]:
            wi = w32_idx[i]
            bo = bit_off[i]
            bits_first = 32 - bo
            low = (w_np[:, wi] >> bo) & ((1 << bits_first) - 1)
            high = w_np[:, wi + 1] & ((1 << (bits - bits_first)) - 1)
            main[:, i] = low | (high << bits_first)

    # Group-expand scales and biases then apply: w = q * scale + bias
    scales_f = scales.float().cpu().numpy()  # (rows, num_groups)
    biases_f = biases.float().cpu().numpy()
    scale_exp = np.repeat(scales_f, group_size, axis=1)  # (rows, in_features)
    bias_exp = np.repeat(biases_f, group_size, axis=1)

    dequant = main.astype(np.float32) * scale_exp + bias_exp
    return torch.from_numpy(dequant).to(torch.bfloat16)


class ModelVariant(StrEnum):
    """Available NexVeridian Qwen3.5-35B-A3B 3-bit MLX model variants."""

    QWEN_3_5_35B_A3B_3BIT = "35B_A3B_3bit"


class ModelLoader(ForgeModel):
    """NexVeridian Qwen3.5-35B-A3B 3-bit MLX model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_35B_A3B_3BIT: LLMModelConfig(
            pretrained_model_name="NexVeridian/Qwen3.5-35B-A3B-3bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_35B_A3B_3BIT

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NexVeridian Qwen3.5-35B-A3B 3-bit MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The checkpoint wraps Qwen3_5MoeForCausalLM under a VL container, so all
        # weight keys carry a "language_model." prefix and weights are MLX 3-bit
        # affine-quantized (uint32 packed, with per-group bfloat16 scales + biases).
        # AutoModelForCausalLM.from_pretrained() cannot handle this, so we load
        # directly from safetensors.

        full_config = AutoConfig.from_pretrained(pretrained_model_name)
        text_config = full_config.text_config

        if self.num_layers is not None:
            text_config.num_hidden_layers = self.num_layers
            if hasattr(text_config, "layer_types"):
                text_config.layer_types = text_config.layer_types[: self.num_layers]

        model_dir = snapshot_download(pretrained_model_name)
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)

        # Read all "language_model.*" tensors from each shard
        PREFIX = "language_model."
        all_raw: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_index["weight_map"].values())):
            shard_path = os.path.join(model_dir, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith(PREFIX):
                        all_raw[key] = sf.get_tensor(key)

        # Build per-tensor bit-width lookup from quantization config overrides.
        # The config stores overrides keyed by "language_model.model.layers.N.mlp.gate"
        # (without the trailing ".weight"); the default is bits=3.
        quant_cfg = full_config.quantization
        default_bits = quant_cfg.get("bits", 3)
        default_group_size = quant_cfg.get("group_size", 64)
        per_tensor_bits: dict[str, int] = {}
        per_tensor_gs: dict[str, int] = {}
        for cfg_key, cfg_val in quant_cfg.items():
            if isinstance(cfg_val, dict):
                per_tensor_bits[cfg_key] = cfg_val.get("bits", default_bits)
                per_tensor_gs[cfg_key] = cfg_val.get("group_size", default_group_size)

        def _get_quant_params(raw_key: str):
            # Strip ".weight" suffix to find the config entry.
            base_key = raw_key.rsplit(".", 1)[0]
            return (
                per_tensor_bits.get(base_key, default_bits),
                per_tensor_gs.get(base_key, default_group_size),
            )

        # Build state dict: strip prefix, dequantize uint32 (MLX) tensors
        SCALES_SUFFIX = ".scales"
        BIASES_SUFFIX = ".biases"
        state_dict: dict[str, torch.Tensor] = {}
        for key, tensor in all_raw.items():
            if key.endswith(SCALES_SUFFIX) or key.endswith(BIASES_SUFFIX):
                continue  # consumed together with the weight tensor below
            model_key = key[len(PREFIX):]
            if tensor.dtype == torch.uint32:
                base = key.rsplit(".", 1)[0]
                scales = all_raw[base + SCALES_SUFFIX]
                biases = all_raw[base + BIASES_SUFFIX]
                bits, gs = _get_quant_params(key)
                tensor = _dequantize_mlx_affine(tensor, scales, biases, bits, gs)
            state_dict[model_key] = tensor

        # Construct on meta device to avoid double-allocating the model weights
        with torch.device("meta"):
            model = Qwen3_5MoeForCausalLM(text_config)
        model = model.to_empty(device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise RuntimeError(
                f"load_state_dict: {len(missing)} missing keys after 3-bit load; "
                f"first 5: {missing[:5]}"
            )
        model = model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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

        inputs["use_cache"] = False

        return inputs

    def _get_text_config(self):
        """Get the text config, handling both nested (MoE) and flat config structures."""
        if hasattr(self.config, "text_config"):
            return self.config.text_config
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
            if hasattr(mlp, "shared_expert"):
                shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", "batch")
                shard_specs[mlp.shared_expert.down_proj.weight] = ("batch", "model")

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
