# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
kaitchup Qwen3.5-9B AutoRound NVFP4 linearattn BF16 model loader implementation for causal language modeling.
"""
import json
import os

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


class ModelVariant(StrEnum):
    """Available kaitchup Qwen3.5-9B AutoRound NVFP4 linearattn BF16 model variants for causal language modeling."""

    QWEN_3_5_9B_AUTOROUND_NVFP4_LINEARATTN_BF16 = "9B_AutoRound_NVFP4_linearattn_BF16"


class ModelLoader(ForgeModel):
    """kaitchup Qwen3.5-9B AutoRound NVFP4 linearattn BF16 model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_9B_AUTOROUND_NVFP4_LINEARATTN_BF16: LLMModelConfig(
            pretrained_model_name="kaitchup/Qwen3.5-9B-autoround-NVFP4-linearattn-BF16",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_9B_AUTOROUND_NVFP4_LINEARATTN_BF16

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
        return ModelInfo(
            model="Qwen3.5-9B AutoRound NVFP4 linearattn BF16",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    @staticmethod
    def _dequantize_nvfp4_weights(model, pretrained_model_name):
        """Load NVFP4-packed weights from safetensors and dequantize to BF16 in-place.

        The safetensors repo stores weight_packed (uint8 NVFP4) instead of weight
        (BF16). Without compressed-tensors awareness, from_pretrained leaves these
        Linear layers randomly initialized. We manually dequantize here.
        """
        from compressed_tensors import unpack_fp4_from_uint8
        from compressed_tensors.quantization import dequantize
        from huggingface_hub import snapshot_download
        from safetensors.torch import safe_open

        model_dir = snapshot_download(pretrained_model_name)

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Collect packed-weight prefixes per shard
        shard_to_prefixes: dict[str, list[str]] = {}
        for key, shard_file in weight_map.items():
            if key.endswith(".weight_packed"):
                prefix = key[: -len(".weight_packed")]
                shard_to_prefixes.setdefault(shard_file, []).append(prefix)

        text_model = model.model  # Qwen3_5TextModel inside ForCausalLM

        for shard_file, prefixes in shard_to_prefixes.items():
            shard_path = os.path.join(model_dir, shard_file)
            with safe_open(shard_path, framework="pt") as f:
                all_keys = set(f.keys())
                for prefix in prefixes:
                    packed = f.get_tensor(f"{prefix}.weight_packed")
                    scale = f.get_tensor(f"{prefix}.weight_scale")
                    gs_key = f"{prefix}.weight_global_scale"
                    global_scale = f.get_tensor(gs_key) if gs_key in all_keys else None

                    # Unpack two NVFP4 nibbles per uint8 byte, then dequantize
                    m, n = packed.shape
                    unpacked = unpack_fp4_from_uint8(packed, m, n * 2)
                    scale_float = scale.to(unpacked.dtype)
                    weight = dequantize(
                        x_q=unpacked,
                        scale=scale_float,
                        global_scale=global_scale,
                        dtype=unpacked.dtype,
                    )

                    # Safetensors prefix: model.language_model.layers.X.mlp.gate_proj
                    # Model path: text_model.layers[X].mlp.gate_proj
                    local_path = prefix.removeprefix("model.language_model.")
                    parts = local_path.split(".")
                    module = text_model
                    for part in parts:
                        module = module[int(part)] if part.isdigit() else getattr(module, part)

                    module.weight.data.copy_(weight.to(module.weight.dtype))

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["device_map"] = "cpu"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # Dequantize NVFP4 weights: safetensors stores weight_packed (uint8) instead
        # of weight (BF16), so from_pretrained leaves quantized layers with random
        # weights. Load and dequantize them manually using compressed-tensors.
        model = self._dequantize_nvfp4_weights(model, pretrained_model_name)

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
            enable_thinking=True,
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

        # Qwen3_5DynamicCache is not a Cache subclass; comparison evaluator
        # calls torch.equal on it and raises TypeError unless we disable caching.
        inputs["use_cache"] = False

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
