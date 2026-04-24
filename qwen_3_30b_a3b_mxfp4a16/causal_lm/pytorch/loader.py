# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NM-Testing Qwen3-30B-A3B MXFP4A16 model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _patch_mxfp4_decompressor():
    """Patch MXFP4PackedCompressor to support E8M0 scale decoding.

    compressed_tensors raises NotImplementedError for MXFP4 decompression because
    the E8M0 uint8 scale requires special handling (value = 2^(byte - 127)).
    This patch implements that conversion so CPU inference is possible.
    """
    try:
        from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        import torch as _torch

        @classmethod
        def decompress(cls, state_dict, scheme):
            state_dict = state_dict.copy()
            packed = state_dict.pop("weight_packed")
            scale = state_dict.get("weight_scale")
            global_scale = state_dict.get("weight_global_scale", None)

            m, n = packed.shape
            unpacked = unpack_fp4_from_uint8(packed, m, n * 2)

            # E8M0 uint8 scale: value = 2^(byte - 127)
            scale_float = _torch.pow(2.0, scale.to(_torch.float32) - 127.0).to(
                unpacked.dtype
            )

            state_dict["weight"] = dequantize(
                x_q=unpacked,
                scale=scale_float,
                global_scale=global_scale,
                dtype=unpacked.dtype,
            )
            state_dict["weight_scale"] = _torch.nn.Parameter(
                scale_float, requires_grad=False
            )
            return state_dict

        MXFP4PackedCompressor.decompress = decompress
    except ImportError:
        pass


_patch_mxfp4_decompressor()

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
    """Available Qwen3-30B-A3B MXFP4A16 model variants for causal language modeling."""

    QWEN_3_30B_A3B_MXFP4A16 = "30B_A3B_MXFP4A16"


class ModelLoader(ForgeModel):
    """NM-Testing Qwen3-30B-A3B MXFP4A16 model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_30B_A3B_MXFP4A16: LLMModelConfig(
            pretrained_model_name="nm-testing/Qwen3-30B-A3B-MXFP4A16",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_30B_A3B_MXFP4A16

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
            model="Qwen3-30B-A3B MXFP4A16",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

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

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
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
