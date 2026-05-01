# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ministral 3 8B model loader implementation for causal language modeling
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, Mistral3ForConditionalGeneration
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Ministral 3 8B model variants."""

    MINISTRAL_3_8B_INSTRUCT_2512_BF16 = "3_8B_Instruct_2512_BF16"
    MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT = "3_8B_Instruct_2512_bnb_4bit"


class ModelLoader(ForgeModel):
    """Ministral 3 8B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BF16: ModelConfig(
            pretrained_model_name="mistralai/Ministral-3-8B-Instruct-2512-BF16",
        ),
        ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT: ModelConfig(
            pretrained_model_name="unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BF16

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ministral",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {
            "padding_side": "right",
        }
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    @staticmethod
    def _dequantize_bnb4_to_bf16(model):
        """Replace all BnB Linear4bit layers with standard bfloat16 Linear layers.

        Params4bit.detach() returns a plain Tensor, which causes
        Parameter.__new__ to raise RuntimeError when model.to(xla_device) is
        called.  Dequantizing to bf16 before device transfer avoids this.
        """
        import bitsandbytes as bnb
        import bitsandbytes.functional as F

        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                if hasattr(module.weight, "quant_state") and module.weight.quant_state is not None:
                    weight_bf16 = F.dequantize_4bit(
                        module.weight.data, module.weight.quant_state
                    ).to(torch.bfloat16)
                else:
                    # Weight already materialized (e.g. CPU load without quant_state)
                    weight_bf16 = module.weight.data.to(torch.bfloat16)
                bias = module.bias
                new_linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=bias is not None,
                    device=weight_bf16.device,
                    dtype=torch.bfloat16,
                )
                new_linear.weight = nn.Parameter(weight_bf16)
                if bias is not None:
                    new_linear.bias = nn.Parameter(bias.to(torch.bfloat16))
                replacements.append((name, new_linear))

        for name, new_module in replacements:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # BnB variants need device_map="cpu" for CPU-based loading
        if self._variant == ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT:
            model_kwargs["device_map"] = "cpu"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.text_config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        # Ministral-3-8B uses Mistral3Config (model_type="mistral3") which requires
        # Mistral3ForConditionalGeneration; AutoModelForCausalLM does not support it.
        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        if self._variant == ModelVariant.MINISTRAL_3_8B_INSTRUCT_2512_BNB_4BIT:
            self._dequantize_bnb4_to_bf16(model)

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        test_input = "How often does the letter r occur in Ministral?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        assert (
            self.config.text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.language_model.layers:
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
