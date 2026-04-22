#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnimateDiff Motion LoRA Tilt Up model loader implementation.

Loads the AnimateDiff pipeline with the Stable Diffusion v1.5 base model,
applies the motion adapter (guoyww/animatediff-motion-adapter-v1-5-2),
and loads the tilt-up motion LoRA weights from
guoyww/animatediff-motion-lora-tilt-up for text-to-video generation
with upward camera tilting.
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter  # type: ignore[import]

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

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_REPO = "guoyww/animatediff-motion-lora-tilt-up"


class ModelVariant(StrEnum):
    """Available AnimateDiff Motion LoRA Tilt Up variants."""

    TILT_UP = "TiltUp"


class ModelLoader(ForgeModel):
    """AnimateDiff Motion LoRA Tilt Up model loader."""

    _VARIANTS = {
        ModelVariant.TILT_UP: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TILT_UP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AnimateDiffPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMATEDIFF_MOTION_LORA_TILT_UP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the AnimateDiff pipeline with motion adapter and tilt-up LoRA.

        Returns:
            AnimateDiffPipeline with motion adapter and LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER,
            torch_dtype=dtype,
        )

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(LORA_REPO)
        self.pipeline.fuse_lora()

        return self.pipeline.unet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare UNet inputs for text-to-video generation with tilt-up motion."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        num_frames = 16
        height = 64
        width = 64
        in_channels = 4
        cross_attention_dim = 768

        sample = torch.randn(
            (batch_size, in_channels, num_frames, height // 8, width // 8),
            dtype=dtype,
        )
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(
            (batch_size * num_frames, 77, cross_attention_dim),
            dtype=dtype,
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output
