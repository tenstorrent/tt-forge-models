# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Inferencerlabs NVIDIA Nemotron 3 Super 120B A12B MLX 4.5-bit model loader
implementation for causal language modeling.

inferencerlabs/NVIDIA-Nemotron-3-Super-120B-A12B-MLX-4.5bit is an MLX-quantized
(4.5-bit) derivative of nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, a
Mixture-of-Experts causal language model with ~120B total parameters and ~12B
active parameters per token. It is exposed as a Nemotron causal LM via Hugging
Face Transformers.
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
    """Available Inferencerlabs NVIDIA Nemotron 3 Super 120B A12B MLX 4.5-bit model variants."""

    NVIDIA_NEMOTRON_3_SUPER_120B_A12B_MLX_4_5BIT = (
        "NVIDIA-Nemotron-3-Super-120B-A12B-MLX-4.5bit"
    )


class ModelLoader(ForgeModel):
    """Inferencerlabs NVIDIA Nemotron 3 Super 120B A12B MLX 4.5-bit model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NVIDIA_NEMOTRON_3_SUPER_120B_A12B_MLX_4_5BIT: LLMModelConfig(
            pretrained_model_name="inferencerlabs/NVIDIA-Nemotron-3-Super-120B-A12B-MLX-4.5bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NVIDIA_NEMOTRON_3_SUPER_120B_A12B_MLX_4_5BIT

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
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="NVIDIA-Nemotron-3-Super-120B-A12B-MLX-4.5bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, trust_remote_code=True, **model_kwargs
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
