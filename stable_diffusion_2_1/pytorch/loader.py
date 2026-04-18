# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2.1 model loader implementation
"""

import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion 2.1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Manojb/stable-diffusion-2-1-base",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Stable Diffusion 2.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        from diffusers import StableDiffusionPipeline

        dtype = dtype_override or torch.float32
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
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

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.float32
        pipe = self._pipeline
        unet = pipe.unet

        prompt = "a photo of an astronaut riding a horse on mars"
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
            (batch_size, in_channels, sample_size, sample_size),
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=torch.long)

        return {
            "sample": sample.to(dtype),
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states.to(dtype),
        }
