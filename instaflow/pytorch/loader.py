# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InstaFlow (XCLiu/instaflow_0_9B_from_sd_1_5) model loader implementation.

InstaFlow is a one-step text-to-image model based on Stable Diffusion 1.5,
distilled using Rectified Flow for single-step image generation.

Available variants:
- INSTAFLOW_0_9B: XCLiu/instaflow_0_9B_from_sd_1_5 text-to-image generation
- RECTIFIED_FLOW_2: XCLiu/2_rectified_flow_from_sd_1_5 few-step text-to-image generation
"""

from typing import Any, Optional

import torch
from diffusers import StableDiffusionPipeline

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
    """Available InstaFlow model variants."""

    INSTAFLOW_0_9B = "InstaFlow_0_9B"
    RECTIFIED_FLOW_2 = "2_rectified_flow_from_sd_1_5"


class ModelLoader(ForgeModel):
    """InstaFlow model loader implementation."""

    _VARIANTS = {
        ModelVariant.INSTAFLOW_0_9B: ModelConfig(
            pretrained_model_name="XCLiu/instaflow_0_9B_from_sd_1_5",
        ),
        ModelVariant.RECTIFIED_FLOW_2: ModelConfig(
            pretrained_model_name="XCLiu/2_rectified_flow_from_sd_1_5",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INSTAFLOW_0_9B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InstaFlow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the InstaFlow pipeline and return the UNet module."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare preprocessed tensor inputs for the UNet."""
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype
        prompt = "a photo of an astronaut riding a horse on mars"

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
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
