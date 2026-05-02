# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit model loader for causal language modeling.
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


def _dequantize_mlx_affine_8bit(raw_sd, group_size=64):
    """Dequantize MLX affine-8bit state dict to standard float tensors.

    mlx-community models store weights as uint32-packed int8 with per-group
    bf16 scales and biases.  Dequant formula: x_float = x_uint8 * scale + bias
    """
    skip = {k for k in raw_sd if k.endswith(".scales") or k.endswith(".biases")}
    result = {}
    for key, tensor in raw_sd.items():
        if key in skip:
            continue
        scales_key = key[: -len(".weight")] + ".scales"
        biases_key = key[: -len(".weight")] + ".biases"
        if (
            key.endswith(".weight")
            and tensor.dtype == torch.uint32
            and scales_key in raw_sd
            and biases_key in raw_sd
        ):
            scales = raw_sd[scales_key]
            biases = raw_sd[biases_key]
            out_f = tensor.shape[0]
            w_u8 = tensor.view(torch.uint8).reshape(out_f, -1)
            in_f = w_u8.shape[1]
            n_grp = in_f // group_size
            sc = scales.float().reshape(out_f, n_grp, 1).expand(-1, -1, group_size).reshape(out_f, in_f)
            bi = biases.float().reshape(out_f, n_grp, 1).expand(-1, -1, group_size).reshape(out_f, in_f)
            result[key] = (w_u8.float() * sc + bi).to(torch.bfloat16)
        else:
            result[key] = tensor
    return result


class ModelVariant(StrEnum):
    """Available lmstudio-community Qwen2.5-Coder-7B-Instruct MLX 8-bit model variants."""

    QWEN_2_5_CODER_7B_INSTRUCT_MLX_8BIT = "7B_Instruct_MLX_8bit"


class ModelLoader(ForgeModel):
    """lmstudio-community Qwen2.5-Coder-7B-Instruct MLX 8-bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-8bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_MLX_8BIT

    sample_text = "write a quick sort algorithm."

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
            model="lmstudio Qwen2.5-Coder-7B-Instruct MLX 8-bit",
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

        # The MLX-quantized config.json has quantization_config without quant_method
        # (only group_size/bits).  Transformers >=5.x raises ValueError on this.
        # Weights are uint32-packed int8 that need manual dequantization.
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import json

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        if self.num_layers is not None:
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers

        # Load sharded safetensors via the index
        index_path = hf_hub_download(pretrained_model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))

        raw_sd = {}
        for shard in shard_files:
            shard_path = hf_hub_download(pretrained_model_name, shard)
            raw_sd.update(load_file(shard_path))

        state_dict = _dequantize_mlx_affine_8bit(raw_sd, group_size=64)

        model_kwargs = {"dtype": dtype}
        model_kwargs.update(kwargs)

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in state dict: {unexpected[:5]}")
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
                "role": "system",
                "content": "You are Qwen, created by TT Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": self.sample_text},
        ]
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

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
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
        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        self.config = config
        return self.config
