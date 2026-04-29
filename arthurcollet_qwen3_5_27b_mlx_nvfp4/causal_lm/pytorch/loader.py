# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
arthurcollet Qwen3.5 27B MLX NVFP4 model loader for causal language modeling.
"""

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

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

# FP4 lookup table (NF4/NVFP4 standard values)
_FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _dequantize_nvfp4(weight_u32: torch.Tensor, scales_u8: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Dequantize NVFP4 uint32 weights + uint8 block scales to bfloat16.

    weight_u32: [..., in_features/8] packed uint32 (8 FP4 nibbles per u32)
    scales_u8:  [..., in_features/group_size] uint8 block-scale exponents
                encoded as exponent = scale_byte - 127 (i.e. scale = 2^(byte-127))
    """
    *prefix, in_packed = weight_u32.shape
    in_features = in_packed * 8  # 8 FP4 values per uint32
    n_groups = in_features // group_size

    # Reinterpret uint32 storage as uint8 bytes (4 bytes per uint32, little-endian)
    weight_u8 = weight_u32.view(torch.uint8)  # [..., in_packed*4]

    lut = torch.tensor(_FP4_VALUES, dtype=torch.bfloat16, device=weight_u32.device)

    # Unpack two FP4 nibbles per byte: lo at even positions, hi at odd
    lo = (weight_u8 & 0x0F).long()  # [..., in_packed*4]
    hi = (weight_u8 >> 4).long()    # [..., in_packed*4]

    out_shape = list(prefix) + [in_features]
    w = torch.empty(out_shape, dtype=torch.bfloat16, device=weight_u32.device)
    w[..., 0::2] = lut[lo]
    w[..., 1::2] = lut[hi]

    # Apply per-group scale: actual_value = w * 2^(scale_byte - 127)
    exponents = scales_u8.to(torch.int32) - 127  # [..., n_groups]
    w_grouped = w.reshape(list(prefix) + [n_groups, group_size])
    w_grouped = torch.ldexp(w_grouped, exponents.unsqueeze(-1).int())

    return w_grouped.reshape(out_shape).to(torch.bfloat16)


def _load_nvfp4_weights(model: Qwen3_5ForCausalLM, snapshot_dir: str, group_size: int = 16):
    """Load NVFP4-quantized safetensors into a Qwen3_5ForCausalLM.

    The checkpoint stores weights with a 'language_model.' prefix (saved from
    Qwen3_5Model which wraps the text model as self.language_model).  We strip
    this prefix and map directly into Qwen3_5ForCausalLM which uses 'model.*'
    keys.  Pairs of (*.weight uint32, *.scales uint8) tensors are dequantized
    to bfloat16; non-quantized tensors (layernorm, biases, etc.) are loaded
    as-is.  Processes one shard at a time to minimize peak RAM overhead.
    """
    import json
    from safetensors import safe_open

    index_file = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    for shard in shard_files:
        path = os.path.join(snapshot_dir, shard)
        # Read all tensors in this shard so weight/scale pairs can be matched
        raw = {}
        with safe_open(path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                raw[key.removeprefix("language_model.")] = sf.get_tensor(key)

        partial = {}
        consumed = set()
        for key, tensor in raw.items():
            if key in consumed:
                continue
            if key.endswith(".weight") and tensor.dtype == torch.uint32:
                scales_key = key[: -len(".weight")] + ".scales"
                if scales_key in raw:
                    partial[key] = _dequantize_nvfp4(tensor, raw[scales_key], group_size)
                    consumed.update({key, scales_key})
            elif key.endswith(".scales"):
                pass  # consumed with its paired .weight above
            else:
                t = (
                    tensor.to(torch.bfloat16)
                    if tensor.dtype in (torch.float32, torch.float16)
                    else tensor
                )
                # MLX stores Conv1d weights as [out, kernel, in]; PyTorch expects [out, in, kernel].
                # Transpose the last two dims when the shape matches the MLX layout.
                if key.endswith("conv1d.weight") and t.ndim == 3 and t.shape[-1] == 1:
                    t = t.transpose(1, 2)
                partial[key] = t
                consumed.add(key)

        model.load_state_dict(partial, strict=False)
        del raw, partial


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

    def _get_snapshot_dir(self) -> str:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            self._variant_config.pretrained_model_name,
            local_files_only=True,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # AutoModelForCausalLM maps model_type='qwen3_5' → Qwen3_5ForCausalLM
        # (uses text_config with the 64-layer SSM hybrid architecture).
        # The checkpoint keys have a 'language_model.' prefix and NVFP4 uint32
        # quantization, so from_pretrained reports all weights as MISSING.
        # We apply the actual weights below.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        full_config = AutoConfig.from_pretrained(pretrained_model_name)
        snapshot_dir = self._get_snapshot_dir()
        _load_nvfp4_weights(
            model,
            snapshot_dir,
            group_size=full_config.quantization.get("group_size", 16),
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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        full_config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        self.config = full_config.text_config
        return self.config
