# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Magistral Small MLX 4-bit model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available Magistral Small MLX 4-bit model variants for causal language modeling."""

    MAGISTRAL_SMALL_2506_MLX_4BIT = "2506_MLX_4bit"


class ModelLoader(ForgeModel):
    """Magistral Small MLX 4-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MAGISTRAL_SMALL_2506_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Magistral-Small-2506-MLX-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAGISTRAL_SMALL_2506_MLX_4BIT

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
            model="Magistral Small MLX 4-bit",
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
    def _dequantize_mlx_4bit(w_uint32, scales, biases, group_size=64):
        """Dequantize MLX 4-bit affine-quantized weight tensor.

        MLX packs 8 unsigned 4-bit values per uint32 (little-endian nibble order).
        Dequant formula: w_float = nibble * scale + bias, per group of group_size elements.
        """
        out_f = w_uint32.shape[0]
        in_f = w_uint32.shape[1] * 8
        n_groups = in_f // group_size

        # Unpack nibbles from uint32 via uint8 view
        w_u8 = w_uint32.view(torch.uint8).reshape(out_f, in_f // 2)
        lo = (w_u8 & 0x0F).to(torch.float32)
        hi = ((w_u8 >> 4) & 0x0F).to(torch.float32)
        # Interleave: even indices get lo nibble, odd indices get hi nibble
        w_nibbles = torch.stack([lo, hi], dim=-1).reshape(out_f, in_f)

        # Expand scales/biases from [out, n_groups] to [out, in]
        scales_exp = (
            scales.float().unsqueeze(-1).expand(out_f, n_groups, group_size).reshape(out_f, in_f)
        )
        biases_exp = (
            biases.float().unsqueeze(-1).expand(out_f, n_groups, group_size).reshape(out_f, in_f)
        )

        return (w_nibbles * scales_exp + biases_exp).to(torch.bfloat16)

    def _load_mlx_4bit_weights(self, model, repo_id, dtype):
        """Load MLX 4-bit safetensors, dequantize quantized linears, and populate model."""
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files

        group_size = 64

        # Collect all safetensor shard file names
        all_files = list(list_repo_files(repo_id))
        shard_files = sorted(f for f in all_files if f.endswith(".safetensors") and "index" not in f)

        # Build raw state dict, collecting (weight, scales, biases) triples
        raw_sd = {}
        for fname in shard_files:
            local = hf_hub_download(repo_id, fname)
            with safe_open(local, framework="pt") as f:
                for k in f.keys():
                    raw_sd[k] = f.get_tensor(k)

        # Separate quantized triples from plain tensors
        quant_bases = {
            k[: -len(".weight")]
            for k in raw_sd
            if k.endswith(".weight") and raw_sd[k].dtype == torch.uint32
        }

        state_dict = {}
        for base in quant_bases:
            w = raw_sd[base + ".weight"]
            sc = raw_sd[base + ".scales"]
            bi = raw_sd[base + ".biases"]
            state_dict[base + ".weight"] = self._dequantize_mlx_4bit(w, sc, bi, group_size)

        for k, v in raw_sd.items():
            if k.endswith(".scales") or k.endswith(".biases"):
                continue
            if k in state_dict:
                continue
            state_dict[k] = v.to(dtype)

        model.load_state_dict(state_dict, strict=False)
        model.tie_weights()
        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.tokenizer is None:
            self._load_tokenizer()

        # Strip quantization_config (MLX format has no quant_method; from_pretrained rejects it)
        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            del config.quantization_config

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype).eval()
        model = self._load_mlx_4bit_weights(model, pretrained_model_name, dtype)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

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
        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        self.config = config
        return self.config
