# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RevAnimated v1.2.2 (danbrown) model loader implementation
"""

from typing import Any, Optional

import torch
from diffusers import StableDiffusionPipeline

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available RevAnimated v1.2.2 (danbrown) model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """RevAnimated v1.2.2 (danbrown) model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="danbrown/RevAnimated-v1-2-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[StableDiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RevAnimated v1.2.2 (danbrown)",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the RevAnimated v1.2.2 pipeline and return the UNet.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        self.pipeline.unet.eval()
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Prepare preprocessed tensor inputs for the UNet.

        Returns:
            list: [latent_sample, timestep, encoder_hidden_states]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        prompt = (
            "masterpiece, best quality, anime style portrait of a young warrior, "
            "vibrant colors, detailed background, cinematic lighting"
        )

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        in_channels = self.pipeline.unet.config.in_channels
        sample_size = self.pipeline.unet.config.sample_size
        latent_sample = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        if dtype_override:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
