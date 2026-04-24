# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimum Intel Internal Testing tiny-stable-diffusion-with-textual-inversion
model loader implementation for text-to-image generation.
"""

import torch
from diffusers import StableDiffusionPipeline
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
    """Available tiny-stable-diffusion-with-textual-inversion model variants."""

    TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION = (
        "tiny_stable_diffusion_with_textual_inversion"
    )


class ModelLoader(ForgeModel):
    """Optimum Intel Internal Testing tiny-stable-diffusion-with-textual-inversion loader."""

    _VARIANTS = {
        ModelVariant.TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-stable-diffusion-with-textual-inversion",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Optimum_Intel_Internal_Testing",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        self._pipeline.to("cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the tiny Stable Diffusion pipeline.

        Returns:
            torch.nn.Module: The UNet model extracted from the pipeline.
        """
        if self._pipeline is None:
            self._load_pipeline()

        unet = self._pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Returns:
            dict: Inputs dict with sample, timestep, and encoder_hidden_states.
        """
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.float32
        pipe = self._pipeline
        unet = pipe.unet

        prompt = ["a photo of an astronaut riding a horse on mars"] * batch_size
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
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states.to(dtype),
        }
