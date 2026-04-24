# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ToonYou Beta 6 model loader implementation
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
    """Available ToonYou Beta 6 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """ToonYou Beta 6 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="frankjoshua/toonyou_beta6",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[StableDiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="ToonYou Beta 6",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ToonYou Beta 6 pipeline."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        """Prepare preprocessed tensor inputs for the UNet."""
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype
        prompt = "masterpiece, best quality, 1girl, cartoon portrait, big eyes, vibrant colors"

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
            1, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
