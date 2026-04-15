# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DreamShaper model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available DreamShaper model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """DreamShaper model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Lykon/DreamShaper",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="DreamShaper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        from diffusers import StableDiffusionPipeline

        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
        )
        self._pipeline.to("cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._pipeline is None:
            self._load_pipeline()

        unet = self._pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None):
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.float32

        pipe = self._pipeline
        unet = pipe.unet

        prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

        in_channels = unet.config.in_channels
        sample_size = unet.config.sample_size
        sample = torch.randn(
            (1, in_channels, sample_size, sample_size),
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=torch.long)

        return {
            "sample": sample.to(dtype),
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states.to(dtype),
        }
