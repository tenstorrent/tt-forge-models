# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Devstral Small MLX 4-bit model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, MistralForCausalLM
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


def _mlx_dequantize(weight, scales, biases, bits=4, group_size=64):
    """Dequantize MLX-format packed-integer weights to bfloat16.

    MLX 4-bit quantization packs (32//bits) unsigned int values per uint32 element,
    LSB first. Each group of `group_size` original values shares one scale and bias:
        original = packed_int4 * scale + bias
    """
    n_per_elem = 32 // bits  # 8 for 4-bit
    out_dim, in_packed = weight.shape
    in_dim = in_packed * n_per_elem

    weight_i32 = weight.view(torch.int32)
    mask = (1 << bits) - 1

    unpacked = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for shift in range(n_per_elem):
        unpacked[:, shift::n_per_elem] = ((weight_i32 >> (shift * bits)) & mask).float()

    scales_f = scales.float().repeat_interleave(group_size, dim=1)
    biases_f = biases.float().repeat_interleave(group_size, dim=1)
    return (unpacked * scales_f + biases_f).to(torch.bfloat16)


def _load_mlx_state_dict(pretrained_model_name, dtype):
    """Load the MLX safetensors checkpoint and dequantize packed int4 weights.

    lmstudio-community MLX safetensors store quantized Linear/embedding weights as
    packed uint32 (8 int4 values per element) with companion '.scales' and '.biases'
    tensors. Standard from_pretrained fails with shape mismatches.
    """
    from huggingface_hub import snapshot_download
    import os
    from safetensors import safe_open

    local_dir = snapshot_download(pretrained_model_name)

    # Collect all safetensors shards
    sf_files = sorted(
        [
            os.path.join(local_dir, f)
            for f in os.listdir(local_dir)
            if f.endswith(".safetensors")
        ]
    )
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {local_dir}")

    # First pass: collect all keys to find quantized bases
    all_keys = []
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt") as f:
            all_keys.extend(f.keys())

    quantized_bases = {
        k[: -len(".weight")]
        for k in all_keys
        if k.endswith(".weight") and (k[: -len(".weight")] + ".scales") in all_keys
    }

    # Second pass: load and dequantize
    state_dict = {}
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(".scales") or key.endswith(".biases"):
                    continue

                tensor = f.get_tensor(key)
                base = key[: -len(".weight")] if key.endswith(".weight") else None

                if base is not None and base in quantized_bases:
                    # Need scales/biases which may be in any shard — load on demand
                    scales = None
                    biases = None
                    for sf2 in sf_files:
                        with safe_open(sf2, framework="pt") as f2:
                            keys2 = list(f2.keys())
                            if base + ".scales" in keys2:
                                scales = f2.get_tensor(base + ".scales")
                            if base + ".biases" in keys2:
                                biases = f2.get_tensor(base + ".biases")
                    tensor = _mlx_dequantize(tensor, scales, biases)
                elif tensor.is_floating_point():
                    tensor = tensor.to(dtype)

                state_dict[key] = tensor

    return state_dict


class ModelVariant(StrEnum):
    """Available Devstral Small MLX 4-bit model variants for causal language modeling."""

    DEVSTRAL_SMALL_2507_MLX_4BIT = "2507_MLX_4bit"


class ModelLoader(ForgeModel):
    """Devstral Small MLX 4-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEVSTRAL_SMALL_2507_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Devstral-Small-2507-MLX-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEVSTRAL_SMALL_2507_MLX_4BIT

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
            model="Devstral Small MLX 4-bit",
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

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Strip the MLX-native quantization_config (no quant_method) so
        # transformers does not raise ValueError for an unknown quantizer.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        # Initialize model architecture on CPU, then load the MLX checkpoint
        # manually to dequantize the packed int4 weights.
        with torch.device("cpu"):
            model = MistralForCausalLM(config)
        model = model.to(dtype)

        state_dict = _load_mlx_state_dict(pretrained_model_name, dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        non_tied_missing = [k for k in missing if "lm_head" not in k]
        if non_tied_missing:
            import warnings
            warnings.warn(f"MLX checkpoint missing keys: {non_tied_missing[:5]}")

        model.tie_weights()
        model.eval()

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

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
