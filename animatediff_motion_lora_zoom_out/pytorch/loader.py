#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnimateDiff Motion LoRA Zoom Out model loader implementation.

Loads the AnimateDiff pipeline with the epiCRealism base model,
applies the motion adapter (guoyww/animatediff-motion-adapter-v1-5-2),
and loads the zoom-out motion LoRA weights from
guoyww/animatediff-motion-lora-zoom-out for text-to-video generation
with zoom-out camera motion.
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter  # type: ignore[import]

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

MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
LORA_REPO = "guoyww/animatediff-motion-lora-zoom-out"


class ModelVariant(StrEnum):
    """Available AnimateDiff Motion LoRA Zoom Out variants."""

    ZOOM_OUT = "ZoomOut"


class ModelLoader(ForgeModel):
    """AnimateDiff Motion LoRA Zoom Out model loader."""

    BASE_MODEL = "emilianJR/epiCRealism"

    _VARIANTS = {
        ModelVariant.ZOOM_OUT: ModelConfig(
            pretrained_model_name=MOTION_ADAPTER,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ZOOM_OUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AnimateDiffPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ANIMATEDIFF_MOTION_LORA_ZOOM_OUT",
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
        """Load the AnimateDiff UNet with motion adapter and zoom-out LoRA.

        Returns:
            UNetMotionModel with LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        adapter = MotionAdapter.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self.BASE_MODEL,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            self.BASE_MODEL,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        self.pipeline.load_lora_weights(LORA_REPO)

        return self.pipeline.unet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare tensor inputs for the UNet forward pass.

        Returns:
            dict with sample, timestep, and encoder_hidden_states tensors.
        """
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
            (batch_size, 77, cross_attention_dim),
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
