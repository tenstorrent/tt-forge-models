# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
nightmedia/OpenAI-gpt-oss-20B-INSTRUCT-Heretic-Uncensored-MXFP4-q8-hi-mlx model loader for causal language modeling.
"""
import json
import re
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
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

# FP4 E2M1 lookup table: index is 4-bit value, result is float
_FP4_E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_affine(packed_u32, scales, biases, group_size):
    """Dequantize 8-bit affine quantized weight from MLX format.

    packed_u32: [..., out, in//4] uint32 (4 uint8 packed per uint32)
    scales: [..., out, in//group_size] bfloat16
    biases: [..., out, in//group_size] bfloat16
    Returns: [..., out, in] bfloat16
    """
    shape = packed_u32.shape[:-1]
    in_features = packed_u32.shape[-1] * 4
    data = packed_u32.view(torch.uint8).reshape(*shape, in_features)
    num_groups = in_features // group_size
    data_g = data.reshape(*shape, num_groups, group_size).float()
    s = scales.reshape(*shape, num_groups, 1).float()
    b = biases.reshape(*shape, num_groups, 1).float()
    return (data_g * s + b).reshape(*shape, in_features).to(torch.bfloat16)


def _dequant_mxfp4(packed_u32, scales_e8m0, group_size):
    """Dequantize 4-bit MXFP4 (E2M1) weight from MLX format.

    packed_u32: [..., out, in//8] uint32 (8 fp4 nibbles packed per uint32)
    scales_e8m0: [..., out, in//group_size] uint8 (exponent-only e8m0 format)
    Returns: [..., out, in] bfloat16
    """
    shape = packed_u32.shape[:-1]
    in_features = packed_u32.shape[-1] * 8
    data_u8 = packed_u32.view(torch.uint8).reshape(*shape, packed_u32.shape[-1] * 4)
    lo = data_u8 & 0xF
    hi = (data_u8 >> 4) & 0xF
    fp4_idx = torch.stack([lo, hi], dim=-1).reshape(*shape, in_features)
    lut = _FP4_E2M1_LUT.to(packed_u32.device)
    data_float = lut[fp4_idx.long()]
    # e8m0 scale: actual_scale = 2^(e8m0 - 127)
    scales_float = torch.exp2(scales_e8m0.float() - 127.0)
    num_groups = in_features // group_size
    data_g = data_float.reshape(*shape, num_groups, group_size)
    s = scales_float.reshape(*shape, num_groups, 1)
    return (data_g * s).reshape(*shape, in_features).to(torch.bfloat16)


def _load_mlx_weights(model_dir, quant_cfg):
    """Load and dequantize MLX-format safetensors into a transformers-compatible state dict.

    MLX uses per-layer quantization with {name}.weight (uint32), {name}.scales, {name}.biases.
    Expert projections (gate_proj, up_proj) must be merged into gate_up_proj.
    """
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_files = sorted(set(idx["weight_map"].values()))

    raw = {}
    for sf in shard_files:
        with safe_open(model_dir / sf, framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)

    default_bits = quant_cfg.get("bits", 4)
    default_mode = quant_cfg.get("mode", "mxfp4")
    default_group_size = quant_cfg.get("group_size", 32)

    def get_layer_quant(base_name):
        cfg = quant_cfg.get(base_name, {})
        return (
            cfg.get("bits", default_bits),
            cfg.get("mode", default_mode),
            cfg.get("group_size", default_group_size),
        )

    def dequantize(base_name, weight):
        bits, mode, group_size = get_layer_quant(base_name)
        scales = raw.get(f"{base_name}.scales")
        biases = raw.get(f"{base_name}.biases")
        if bits == 8 and mode == "affine" and scales is not None and biases is not None:
            return _dequant_affine(weight, scales, biases, group_size)
        if bits == 4 and mode == "mxfp4" and scales is not None:
            return _dequant_mxfp4(weight, scales, group_size)
        return weight.to(torch.bfloat16)

    state_dict = {}
    expert_gate = {}
    expert_up = {}
    expert_down = {}

    for key, tensor in raw.items():
        # Drop quantization artifacts
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        if key.endswith(".weight") and tensor.dtype == torch.uint32:
            base = key[: -len(".weight")]
            deq = dequantize(base, tensor)

            if ".mlp.experts.gate_proj" in key:
                layer_match = re.search(r"model\.layers\.(\d+)", key)
                if layer_match:
                    expert_gate[int(layer_match.group(1))] = deq
            elif ".mlp.experts.up_proj" in key:
                layer_match = re.search(r"model\.layers\.(\d+)", key)
                if layer_match:
                    expert_up[int(layer_match.group(1))] = deq
            elif ".mlp.experts.down_proj" in key:
                layer_match = re.search(r"model\.layers\.(\d+)", key)
                if layer_match:
                    expert_down[int(layer_match.group(1))] = deq
            else:
                state_dict[key] = deq
        elif key.endswith(".bias") and not key.endswith(".biases"):
            # Regular linear bias; skip expert biases (handled separately)
            if ".mlp.experts." not in key:
                state_dict[key] = tensor
        else:
            # Unquantized tensors (RMSNorm weights, sinks, etc.)
            state_dict[key] = tensor

    # Merge gate_proj + up_proj -> gate_up_proj and handle expert biases
    layer_nums = (
        set(expert_gate.keys()) | set(expert_up.keys()) | set(expert_down.keys())
    )
    for n in sorted(layer_nums):
        prefix = f"model.layers.{n}.mlp.experts"
        gate_deq = expert_gate.get(n)
        up_deq = expert_up.get(n)
        down_deq = expert_down.get(n)

        if gate_deq is not None and up_deq is not None:
            # MLX: [num_experts, out=intermediate, in=hidden]
            # Transformers gate_up_proj: [num_experts, in=hidden, 2*out=2*intermediate]
            state_dict[f"{prefix}.gate_up_proj"] = torch.cat(
                [gate_deq.permute(0, 2, 1), up_deq.permute(0, 2, 1)], dim=-1
            )
            gate_bias = raw.get(f"{prefix}.gate_proj.bias")
            up_bias = raw.get(f"{prefix}.up_proj.bias")
            if gate_bias is not None and up_bias is not None:
                state_dict[f"{prefix}.gate_up_proj_bias"] = torch.cat(
                    [gate_bias, up_bias], dim=-1
                )

        if down_deq is not None:
            # MLX: [num_experts, out=hidden, in=intermediate]
            # Transformers down_proj: [num_experts, in=intermediate, out=hidden]
            state_dict[f"{prefix}.down_proj"] = down_deq.permute(0, 2, 1)
            down_bias = raw.get(f"{prefix}.down_proj.bias")
            if down_bias is not None:
                state_dict[f"{prefix}.down_proj_bias"] = down_bias

    return state_dict


class ModelVariant(StrEnum):
    """Available nightmedia OpenAI GPT-OSS 20B INSTRUCT Heretic Uncensored MLX model variants."""

    OPENAI_GPT_OSS_20B_INSTRUCT_HERETIC_UNCENSORED_MXFP4_Q8_HI_MLX = (
        "OPENAI_GPT_OSS_20B_INSTRUCT_HERETIC_UNCENSORED_MXFP4_Q8_HI_MLX"
    )


class ModelLoader(ForgeModel):
    """nightmedia OpenAI GPT-OSS 20B INSTRUCT Heretic Uncensored MLX model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.OPENAI_GPT_OSS_20B_INSTRUCT_HERETIC_UNCENSORED_MXFP4_Q8_HI_MLX: LLMModelConfig(
            pretrained_model_name="nightmedia/OpenAI-gpt-oss-20B-INSTRUCT-Heretic-Uncensored-MXFP4-q8-hi-mlx",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = (
        ModelVariant.OPENAI_GPT_OSS_20B_INSTRUCT_HERETIC_UNCENSORED_MXFP4_Q8_HI_MLX
    )

    sample_text = "What is your favorite city?"

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
            model="nightmedia OpenAI GPT-OSS 20B INSTRUCT Heretic Uncensored MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        self.load_config()

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Download model files (metadata + weights)
        model_dir = snapshot_download(pretrained_model_name)

        # Build model architecture with random init, then load dequantized weights
        model = AutoModelForCausalLM.from_config(
            self.config,
            attn_implementation="eager",
            torch_dtype=target_dtype,
        )

        # The config stores the raw MLX quantization dict under the 'quantization' key
        raw_config = json.loads((Path(model_dir) / "config.json").read_text())
        quant_cfg = raw_config.get("quantization", {})

        state_dict = _load_mlx_weights(model_dir, quant_cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            import warnings

            warnings.warn(f"Missing keys when loading MLX weights: {missing[:10]}")

        model = model.to(target_dtype).eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        # The hub config has an MLX quantization_config without a quant_method.
        # transformers checks hasattr(config, "quantization_config"), so we must
        # delete the attribute entirely to prevent the quantizer from activating.
        if hasattr(self.config, "quantization_config"):
            delattr(self.config, "quantization_config")
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers
        return self.config
