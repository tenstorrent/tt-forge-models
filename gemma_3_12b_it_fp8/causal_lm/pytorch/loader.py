# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
pytorch/gemma-3-12b-it-FP8 model loader implementation for causal language modeling.
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
    """Available Gemma 3 12B IT FP8 model variants for causal language modeling."""

    GEMMA_3_12B_IT_FP8 = "12B_IT_FP8"


class ModelLoader(ForgeModel):
    """pytorch/gemma-3-12b-it-FP8 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_12B_IT_FP8: LLMModelConfig(
            pretrained_model_name="pytorch/gemma-3-12b-it-FP8",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_12B_IT_FP8

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
            model="Gemma 3 12B IT FP8",
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

        config = AutoConfig.from_pretrained(pretrained_model_name)
        # Strip FP8 quantization_config so weights load as plain BF16 — CPU
        # _scaled_mm only supports per-tensor scaling, not the per-channel
        # scaling used by this model's torchao float8 quantization.
        cfg = getattr(config, "text_config", config)
        cfg.quantization_config = None

        if self.num_layers is not None:
            cfg.num_hidden_layers = self.num_layers

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        # torchao's safetensors deserializer wraps FP8 checkpoint tensors in
        # Float8Tensor. model.to() doesn't dequantize them; use the class's
        # own dequantize() which applies the stored per-row scale correctly.
        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        try:
            from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
                Float8Tensor as TorchAoFloat8Tensor,
            )

            with torch.no_grad():
                for module in model.modules():
                    for pname in list(module._parameters.keys()):
                        param = module._parameters[pname]
                        if param is not None and isinstance(
                            param.data, TorchAoFloat8Tensor
                        ):
                            module._parameters[pname] = torch.nn.Parameter(
                                param.data.dequantize(output_dtype=target_dtype)
                            )
        except ImportError:
            model = model.to(target_dtype)

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
