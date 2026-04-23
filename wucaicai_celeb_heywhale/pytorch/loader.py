# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wucaicai Celeb HeyWhale model loader implementation
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
from diffusers import StableDiffusionPipeline


class ModelVariant(StrEnum):
    """Available Wucaicai Celeb HeyWhale model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Wucaicai Celeb HeyWhale model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="rootkan/wucaicai-celeb-heywhale",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    prompt = "a photo of wucaicai celeb tokyo walk"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Wucaicai Celeb HeyWhale",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Wucaicai Celeb HeyWhale pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet2DConditionModel from the pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return preprocessed tensor inputs for the UNet.

        Args:
            dtype_override: Optional torch.dtype to override the input tensors' dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            list: Input tensors [latent_sample, timestep, encoder_hidden_states].
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        text_inputs = self.pipeline.tokenizer(
            [self.prompt] * batch_size,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        latent_sample = torch.randn(
            batch_size,
            self.pipeline.unet.config.in_channels,
            64,
            64,
            dtype=dtype,
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        if dtype_override:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]
