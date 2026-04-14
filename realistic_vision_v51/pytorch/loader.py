# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Realistic Vision v5.1 model loader implementation
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
    """Available Realistic Vision v5.1 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Realistic Vision v5.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stablediffusionapi/realistic-vision-v51",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Realistic Vision v5.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype=None):
        from diffusers import StableDiffusionPipeline

        dtype = dtype or torch.bfloat16
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self._pipeline.to("cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Realistic Vision v5.1 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The UNet component of the pipeline.
        """
        if self._pipeline is None:
            self._load_pipeline(dtype=dtype_override)

        unet = self._pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Realistic Vision v5.1 UNet.

        Args:
            dtype_override: Optional dtype for the inputs.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary of UNet input tensors.
        """
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.bfloat16
        pipe = self._pipeline
        unet = pipe.unet

        prompt = "RAW photo, a portrait of a woman in a rustic setting, 8k uhd, high quality, film grain"
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
