#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VACE-Wan2.1-1.3B-Preview model loader implementation.

Loads the Wan-AI/Wan2.1-VACE-1.3B-diffusers model and returns the
WanVACETransformer3DModel for compilation and inference benchmarking.
"""

from typing import Any, Optional

import torch

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
    """Available VACE-Wan2.1-1.3B-Preview variants."""

    VACE_WAN_2_1_1_3B_PREVIEW = "VACE-Wan2.1-1.3B-Preview"


class ModelLoader(ForgeModel):
    """VACE-Wan2.1-1.3B-Preview model loader for reference-to-video generation."""

    _VARIANTS = {
        ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VACE_WAN_2_1_1_3B_PREVIEW

    DEFAULT_PROMPT = (
        "A character walking gracefully across a sunlit garden, "
        "smooth animation, detailed motion, cinematic lighting"
    )

    # VAE compression factors for Wan2.1: spatial=8x, temporal=4x
    _VAE_SPATIAL_SCALE = 8
    _VAE_TEMPORAL_SCALE = 4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VACE-Wan2.1-1.3B-Preview",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> None:
        from diffusers import WanVACEPipeline

        self.pipeline = WanVACEPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the WanVACETransformer3DModel."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype=dtype)
        return self.pipeline.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic inputs for the WanVACETransformer3DModel."""
        dtype = torch.bfloat16
        if self.pipeline is None:
            self._load_pipeline(dtype)

        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT

        # Encode text prompt using the pipeline's tokenizer and text encoder
        text_inputs = self.pipeline.tokenizer(
            prompt_value,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype=dtype)

        # Latent shape: [batch, in_channels, frames, h, w] with small spatial dims
        in_channels = self.pipeline.transformer.config.in_channels
        latents = torch.randn(1, in_channels, 2, 8, 8, dtype=dtype)
        timestep = torch.tensor([500], dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }
