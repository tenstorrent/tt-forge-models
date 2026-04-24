# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MoringLabs Nemotron 3 Super 120B A12B MLX 3.6-bit model loader implementation for causal language modeling.
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
    """Available MoringLabs Nemotron 3 Super 120B A12B MLX 3.6-bit model variants for causal language modeling."""

    NEMOTRON_3_SUPER_120B_A12B_MLX_3_6BIT = "3_Super_120B_A12B_MLX_3_6bit"


class ModelLoader(ForgeModel):
    """MoringLabs Nemotron 3 Super 120B A12B MLX 3.6-bit model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_3_SUPER_120B_A12B_MLX_3_6BIT: LLMModelConfig(
            pretrained_model_name="MoringLabs/Nemotron-3-Super-120B-A12B-MLX-3.6bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_3_SUPER_120B_A12B_MLX_3_6BIT

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
            model="MoringLabs Nemotron 3 Super 120B A12B MLX 3.6-bit",
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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _load_config(self):
        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        # The MLX quantization_config is a plain dict without a quant_method field,
        # which transformers' quantizer registry does not recognise. Clear it so that
        # from_pretrained and random-weight instantiation both proceed without error.
        if isinstance(getattr(config, "quantization_config", None), dict):
            config.quantization_config = None
        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._load_config()

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {"config": config, "ignore_mismatched_sizes": True}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, trust_remote_code=True, **model_kwargs
            )

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

    def load_config(self):
        self.config = self._load_config()
        return self.config
