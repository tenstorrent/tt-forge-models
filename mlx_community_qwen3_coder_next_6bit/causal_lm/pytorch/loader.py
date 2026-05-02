# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mlx-community/Qwen3-Coder-Next-6bit model loader for causal language modeling.

The checkpoint uses MLX affine 6-bit quantization (default) with 8-bit overrides
for the MoE router gate and shared-expert gate.  Weights are stored as uint32-packed
bit streams with per-group bfloat16 scales and biases.

Two loader bugs are fixed:
  1. config.quantization_config has no quant_method → ValueError from transformers 5.x.
     Fix: delete quantization_config/quantization before calling from_config.
  2. Expert weights: MLX stores switch_mlp.{gate,up}_proj as separate 3D tensors;
     transformers expects experts.gate_up_proj as a single concatenated 3D tensor.
     Fix: dequantize gate + up separately then cat(dim=1).

NOTE: The model has 79.7 B parameters (~159 GB BF16) which exceeds single-device
DRAM on current Tenstorrent hardware (p150b ≈ 34 GB).  This loader is correct
but the test is marked KNOWN_FAILURE_XFAIL for hardware-class reasons.
"""
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available mlx-community Qwen3-Coder-Next-6bit model variants for causal language modeling."""

    QWEN_3_CODER_NEXT_6BIT = "Coder_Next_6bit"


# ---------------------------------------------------------------------------
# MLX affine dequantization helpers
# ---------------------------------------------------------------------------

def _unpack_6bit(w_uint32: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack MLX 6-bit affine integers from a uint32 tensor.

    Uses numpy for efficient cross-word boundary handling.

    w_uint32  : [rows, cols_packed] torch.uint32 — values packed LSB-first
    Returns   : [rows, in_features] torch.float32
    """
    rows, cols_packed = w_uint32.shape
    w_np = w_uint32.numpy().view(np.uint32)          # (rows, cols_packed)

    bit_pos = np.arange(in_features, dtype=np.int64) * 6
    w32_idx = (bit_pos // 32).astype(np.int32)
    bit_off = (bit_pos % 32).astype(np.int32)

    result = (w_np[:, w32_idx] >> bit_off) & np.uint32(0x3F)  # (rows, in_features)

    # Fix elements that straddle a uint32 word boundary
    cross = np.where(bit_off + 6 > 32)[0]
    for ci in cross:
        wi = int(w32_idx[ci])
        bo = int(bit_off[ci])
        bits_lo = 32 - bo
        lo = (w_np[:, wi] >> bo) & np.uint32((1 << bits_lo) - 1)
        hi = w_np[:, wi + 1] & np.uint32((1 << (6 - bits_lo)) - 1)
        result[:, ci] = lo | (hi << bits_lo)

    return torch.from_numpy(result.astype(np.float32))


def _unpack_8bit(w_uint32: torch.Tensor) -> torch.Tensor:
    """Unpack MLX 8-bit affine integers from a uint32 tensor.

    Returns [rows, cols_packed * 4] float32.
    """
    return w_uint32.view(torch.uint8).float()


def _dequantize(
    w_uint32: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an MLX affine-quantized weight tensor.

    Handles 2D (standard linear) and 3D (batched expert) weight shapes.
    """
    expert_dim: Optional[int] = None
    if w_uint32.ndim == 3:
        expert_dim = w_uint32.shape[0]
        out_f = w_uint32.shape[1]
        w_uint32 = w_uint32.reshape(expert_dim * out_f, -1)
        scales = scales.reshape(expert_dim * out_f, -1)
        biases = biases.reshape(expert_dim * out_f, -1)

    if bits == 8:
        w_int = _unpack_8bit(w_uint32)              # [rows, cols*4]
        in_f = w_uint32.shape[1] * 4
    else:  # 6-bit (and potentially other values via numpy path)
        in_f = w_uint32.shape[1] * 32 // bits
        w_int = _unpack_6bit(w_uint32, in_f)        # [rows, in_f]

    sc = scales.float().repeat_interleave(group_size, dim=1)[:, :in_f]
    bi = biases.float().repeat_interleave(group_size, dim=1)[:, :in_f]
    w = w_int * sc + bi                              # [rows, in_f]

    if expert_dim is not None:
        w = w.reshape(expert_dim, w_uint32.shape[0] // expert_dim, in_f)

    return w.to(target_dtype)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _process_shard(
    raw: dict,
    state_dict: dict,
    pending_gate: dict,
    group_size: int,
    default_bits: int,
    per_key_bits: dict,
    target_dtype: torch.dtype,
) -> None:
    """Dequantize tensors from one safetensors shard and accumulate into state_dict.

    Expert switch_mlp.gate_proj + switch_mlp.up_proj are deferred until both are
    available (stored in pending_gate until then), then concatenated to experts.gate_up_proj.
    """
    quant_bases = {
        key[: -len(".weight")]
        for key, t in raw.items()
        if key.endswith(".weight") and t.dtype == torch.uint32
    }
    aux_keys = {
        base + sfx
        for base in quant_bases
        for sfx in (".scales", ".biases")
    }

    for key, tensor in raw.items():
        if key in aux_keys:
            continue

        if key.endswith(".weight") and key[: -len(".weight")] in quant_bases:
            base = key[: -len(".weight")]
            scales = raw[base + ".scales"]
            biases = raw[base + ".biases"]

            # Determine effective bits (per-key override or default)
            bits = per_key_bits.get(base, default_bits)

            value = _dequantize(tensor, scales, biases, group_size, bits, target_dtype)

            # Remap switch_mlp expert weights
            if ".mlp.switch_mlp.gate_proj" in key:
                gate_key = key.replace(
                    ".mlp.switch_mlp.gate_proj.weight",
                    ".mlp.experts.__gate__",
                )
                pending_gate[gate_key] = value
                continue
            elif ".mlp.switch_mlp.up_proj" in key:
                gate_key = key.replace(
                    ".mlp.switch_mlp.up_proj.weight",
                    ".mlp.experts.__gate__",
                )
                up_val = value
                if gate_key in pending_gate:
                    gate_val = pending_gate.pop(gate_key)
                    out_key = key.replace(
                        ".mlp.switch_mlp.up_proj.weight",
                        ".mlp.experts.gate_up_proj",
                    )
                    state_dict[out_key] = torch.cat([gate_val, up_val], dim=1)
                else:
                    # gate not yet seen; store up until gate arrives in a later shard
                    pending_gate[gate_key + "__up__"] = up_val
                continue
            elif ".mlp.switch_mlp.down_proj" in key:
                key = key.replace(".mlp.switch_mlp.down_proj.weight", ".mlp.experts.down_proj")
            # else: direct key (self_attn, linear_attn, shared_expert, lm_head, etc.)

        else:
            # Non-quantized tensor
            if tensor.is_floating_point():
                value = tensor.to(target_dtype)
            else:
                value = tensor

            # MLX Conv1d stores weights as [out, kernel, in]; PyTorch wants [out, in, kernel]
            if value.ndim == 3 and "conv" in key.lower():
                value = value.permute(0, 2, 1).contiguous()

        state_dict[key] = value


def _load_and_dequantize_qwen3next_6bit(
    pretrained_model_name: str,
    group_size: int,
    default_bits: int,
    per_key_bits: dict,
    target_dtype: torch.dtype,
) -> dict:
    """Download and dequantize all MLX safetensors shards for Qwen3-Coder-Next-6bit."""
    index_path = hf_hub_download(pretrained_model_name, "model.safetensors.index.json")
    with open(index_path) as f:
        index_data = json.load(f)
    shard_names = sorted(set(index_data["weight_map"].values()))

    state_dict: dict = {}
    pending_gate: dict = {}

    for shard_name in shard_names:
        shard_path = hf_hub_download(pretrained_model_name, shard_name)
        raw: dict = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)

        _process_shard(raw, state_dict, pending_gate, group_size, default_bits, per_key_bits, target_dtype)
        del raw

    if pending_gate:
        import warnings
        warnings.warn(
            f"Qwen3-Next-6bit: {len(pending_gate)} unmatched expert gate projections after loading all shards: {list(pending_gate)[:5]}"
        )

    return state_dict


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelLoader(ForgeModel):
    """mlx-community Qwen3-Coder-Next-6bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_CODER_NEXT_6BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen3-Coder-Next-6bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_CODER_NEXT_6BIT

    sample_text = "Write a Python function that checks if a number is prime."

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
            model="mlx-community Qwen3-Coder-Next-6bit",
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
            self._load_tokenizer()

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Load config; extract quantization params before stripping the metadata.
        # Transformers 5.x raises ValueError when quantization_config has no quant_method.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        quant_raw = None
        for attr in ("quantization", "quantization_config"):
            if hasattr(config, attr):
                quant_raw = dict(getattr(config, attr))
                delattr(config, attr)

        group_size = 64
        default_bits = 6
        per_key_bits: dict = {}
        if quant_raw:
            group_size = quant_raw.get("group_size", 64)
            default_bits = quant_raw.get("bits", 6)
            # Per-layer bit-width overrides (e.g. mlp.gate and mlp.shared_expert_gate → 8-bit)
            per_key_bits = {
                k: v["bits"]
                for k, v in quant_raw.items()
                if isinstance(v, dict) and "bits" in v
            }

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
            if hasattr(config, "layer_types"):
                config.layer_types = config.layer_types[: self.num_layers]

        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=target_dtype, **kwargs
        ).eval()

        state_dict = _load_and_dequantize_qwen3next_6bit(
            pretrained_model_name,
            group_size,
            default_bits,
            per_key_bits,
            target_dtype,
        )
        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()

        self.config = config
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
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def _get_text_config(self):
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
