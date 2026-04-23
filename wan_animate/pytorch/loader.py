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

        self._transformer = self.pipeline.transformer
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the Wan Animate transformer."""
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # WanAnimateTransformer3DModel config: in_channels=36, text_dim=4096
        in_channels = 36
        text_dim = 4096
        txt_seq_len = 32

        # Small spatial/temporal latent dimensions for testing
        # patch_size = (1, 2, 2), so spatial dims are halved
        frame, height, width = 1, 4, 4
        seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        timestep = torch.tensor([500] * batch_size, dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }
