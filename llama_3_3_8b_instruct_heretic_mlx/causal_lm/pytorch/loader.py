# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.3 8B Instruct Heretic MLX model loader implementation for causal language modeling.
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
    """Available Llama 3.3 8B Instruct Heretic MLX model variants for causal language modeling."""

    LLAMA_3_3_8B_INSTRUCT_HERETIC_MLX_8BIT = "3.3_8B_Instruct_Heretic_MLX_8Bit"


class ModelLoader(ForgeModel):
    """Llama 3.3 8B Instruct Heretic MLX model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_3_8B_INSTRUCT_HERETIC_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="coderavi/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning-mlx-8Bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_3_8B_INSTRUCT_HERETIC_MLX_8BIT

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
        return ModelInfo(
            model="Llama 3.3 8B Instruct Heretic MLX",
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
        # (only group_size/bits/mode).  Transformers >=5.x raises ValueError on this.
        # Also, weights are uint32-packed int8 that need manual dequantization.
        # Transformers 5.x also forbids passing state_dict with a model name string,
        # so we use from_config + load_state_dict.
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        import json

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

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
                "role": "user",
                "content": self.sample_text,
            }
        ]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
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
        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        self.config = config
        return self.config
