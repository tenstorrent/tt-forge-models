# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-R1-Distill-Qwen-7B 4-bit model loader implementation for causal language modeling.
"""
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
    """Available DeepSeek-R1-Distill-Qwen-7B 4-bit model variants for causal language modeling."""

    DEEPSEEK_R1_DISTILL_QWEN_7B_4BIT = "DeepSeek_R1_Distill_Qwen_7B_4bit"


class ModelLoader(ForgeModel):
    """DeepSeek-R1-Distill-Qwen-7B 4-bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_7B_4BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_7B_4BIT

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
            model="DeepSeek-R1-Distill-Qwen-7B 4-bit",
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

    @staticmethod
    def _mlx_dequantize(packed_weight, scales, biases, bits=4, group_size=64):
        """Dequantize MLX packed-integer weights to bfloat16.

        MLX stores weights as uint32 with (32//bits) int4 values packed LSB-first.
        Each group of `group_size` elements shares one scale and one bias:
            original = packed_int4 * scale + bias
        """
        n_per_elem = 32 // bits  # 8 for 4-bit
        out_dim, in_packed = packed_weight.shape
        in_dim = in_packed * n_per_elem

        weight_i32 = packed_weight.view(torch.int32)
        mask = (1 << bits) - 1

        unpacked = torch.zeros(out_dim, in_dim, dtype=torch.float32)
        for shift in range(n_per_elem):
            unpacked[:, shift::n_per_elem] = (
                (weight_i32 >> (shift * bits)) & mask
            ).float()

        scales_f = scales.float().repeat_interleave(group_size, dim=1)
        biases_f = biases.float().repeat_interleave(group_size, dim=1)
        return (unpacked * scales_f + biases_f).to(torch.bfloat16)

    @staticmethod
    def _load_mlx_state_dict(pretrained_model_name, target_dtype):
        """Load safetensors from mlx-community repo and dequantize MLX 4-bit weights.

        MLX stores quantized linear/embedding weights as uint32 packed int4 tensors
        alongside companion .scales and .biases tensors.  This function dequantizes
        those packed weights to bfloat16 so they can be loaded into a standard
        transformers Qwen2 model.  Plain float tensors (layer norms, attention biases)
        are cast to target_dtype and passed through unchanged.
        """
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        local_dir = snapshot_download(pretrained_model_name)
        sf_path = os.path.join(local_dir, "model.safetensors")

        with safe_open(sf_path, framework="pt") as f:
            all_keys = list(f.keys())

        quantized_bases = {
            k[: -len(".weight")]
            for k in all_keys
            if k.endswith(".weight")
            and (k[: -len(".weight")] + ".scales") in all_keys
        }

        state_dict = {}
        with safe_open(sf_path, framework="pt") as f:
            for key in all_keys:
                if key.endswith(".scales") or key.endswith(".biases"):
                    continue
                base = key[: -len(".weight")] if key.endswith(".weight") else None
                if base is not None and base in quantized_bases:
                    packed = f.get_tensor(key)
                    scales = f.get_tensor(base + ".scales")
                    biases_q = f.get_tensor(base + ".biases")
                    state_dict[key] = ModelLoader._mlx_dequantize(
                        packed, scales, biases_q
                    )
                else:
                    tensor = f.get_tensor(key)
                    state_dict[key] = tensor.to(target_dtype)

        return state_dict

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Load config and remove the MLX quantization_config dict, which lacks the
        # `quant_method` field that transformers 5.x requires for HF quantizers.
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.quantization_config = None

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config).to(target_dtype).eval()

        state_dict = self._load_mlx_state_dict(pretrained_model_name, target_dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            import warnings
            warnings.warn(f"Missing keys when loading MLX weights: {missing[:5]}{'...' if len(missing) > 5 else ''}")

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
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.config
