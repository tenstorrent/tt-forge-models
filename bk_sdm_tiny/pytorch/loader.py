# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BK-SDM-Tiny model loader implementation.

Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM) is an
architecturally compressed SDM for efficient text-to-image synthesis. BK-SDM-Tiny
removes residual and attention blocks from the U-Net of Stable Diffusion v1.4
and is loaded via the standard StableDiffusionPipeline.
"""

import torch
from typing import Any, Optional

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
    """Available BK-SDM-Tiny model variants."""

    BK_SDM_TINY = "bk-sdm-tiny"


class ModelLoader(ForgeModel):
    """BK-SDM-Tiny model loader implementation."""

    _VARIANTS = {
        ModelVariant.BK_SDM_TINY: ModelConfig(
            pretrained_model_name="nota-ai/bk-sdm-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BK_SDM_TINY

    prompt = "a tropical bird sitting on a branch of a tree"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[StableDiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BK-SDM-Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the BK-SDM-Tiny pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the BK-SDM-Tiny UNet model.

        Args:
            dtype_override: Optional torch.dtype to override input tensor dtype.

        Returns:
            list: [latent_sample, timestep, encoder_hidden_states]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        text_inputs = self.pipeline.tokenizer(
            self.prompt,
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
            1, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        if dtype_override is not None:
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
