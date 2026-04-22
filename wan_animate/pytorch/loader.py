#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Animate diffusion model loader implementation.

Supports text-to-video animation generation using the Wan 2.2 Animate 14B model.

Available variants:
- WAN22_ANIMATE_14B: Wan 2.2 Animate 14B (text-to-video animation)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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

# hidden_states: (B, 2*latent_channels+4, T+1, H, W)
# pose_hidden_states: (B, latent_channels, T, H, W)
# With T+1=3 frames, H=W=4, patch_size=(1,2,2), the post-patch seq len S = 3*2*2 = 12.
# face_pixel_values: (B, 3, T_face, 512, 512); with T_face=2 the face encoder outputs T_out=1
# frame, then after padding T=T_out+1=2, and S/T=12/2=6 is an integer as required.
TRANSFORMER_NUM_FRAMES = 2  # T in pose; hidden gets T+1=3
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8
FACE_FRAME_SIZE = 512  # motion encoder requires exactly this spatial resolution
FACE_NUM_FRAMES = 2


class ModelVariant(StrEnum):
    """Available Wan Animate model variants."""

    WAN22_ANIMATE_14B = "2.2_Animate_14B"


class ModelLoader(ForgeModel):
    """Wan 2.2 Animate diffusion model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_ANIMATE_14B: ModelConfig(
            pretrained_model_name="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_ANIMATE_14B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WAN_ANIMATE",
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
        if self._transformer is not None:
            if dtype_override is not None:
                self._transformer = self._transformer.to(dtype=dtype_override)
            return self._transformer

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        self._transformer = pipeline.transformer
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        if self._transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self._transformer.config

        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES + 1,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "pose_hidden_states": torch.randn(
                1,
                config.latent_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "face_pixel_values": torch.randn(
                1,
                3,
                FACE_NUM_FRAMES,
                FACE_FRAME_SIZE,
                FACE_FRAME_SIZE,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }
