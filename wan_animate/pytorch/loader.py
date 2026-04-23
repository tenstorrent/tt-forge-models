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
from diffusers import WanAnimatePipeline, WanAnimateTransformer3DModel

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

    DEFAULT_PROMPT = (
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanAnimatePipeline] = None
        self._transformer: Optional[WanAnimateTransformer3DModel] = None

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
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ) -> WanAnimateTransformer3DModel:
        """Load and return the Wan Animate transformer (a torch.nn.Module)."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self.pipeline = WanAnimatePipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        self._transformer = self.pipeline.transformer.to(dtype=dtype)
        self._transformer.eval()
        return self._transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the Wan Animate transformer."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        # WanAnimateTransformer3DModel config dims:
        #   in_channels=36 (16 noisy latent + 16 reference + 4 mask)
        #   text_dim=4096, image_dim=1280, patch_size=(1,2,2)
        # Attention processor hardcodes txt context length of 512.
        in_channels = 36
        text_dim = 4096
        txt_seq_len = 512
        image_dim = 1280
        img_seq_len = 32

        # Small spatial/temporal dims: T latent frames, H x W spatial
        # hidden_states shape: (B, in_channels, T+1, H, W) — T+1 because of
        # a prepended reference frame; pose_hidden_states uses T frames.
        num_latent_frames = 1
        lat_h, lat_w = 4, 4

        hidden_states = torch.randn(
            batch_size, in_channels, num_latent_frames + 1, lat_h, lat_w, dtype=dtype
        )
        pose_hidden_states = torch.randn(
            batch_size, 16, num_latent_frames, lat_h, lat_w, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_image = torch.randn(
            batch_size, img_seq_len, image_dim, dtype=dtype
        )
        # face_pixel_values: (B, C, S, H', W') — S face frames in pixel space
        face_pixel_values = torch.randn(
            batch_size, 3, num_latent_frames, 112, 112, dtype=dtype
        )
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_image": encoder_hidden_states_image,
            "pose_hidden_states": pose_hidden_states,
            "face_pixel_values": face_pixel_values,
            "timestep": timestep,
            "return_dict": False,
        }
