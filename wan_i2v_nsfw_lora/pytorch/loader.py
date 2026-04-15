#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 I2V NSFW LoRA model loader implementation.

Loads the Wan 2.2 I2V base pipeline and applies NSFW LoRA weights
from lopi999/Wan2.2-I2V_General-NSFW-LoRA for image-to-video generation.

Available variants:
- WAN22_I2V_NSFW_HIGH: High-rank LoRA variant
- WAN22_I2V_NSFW_LOW: Low-rank LoRA variant
"""

from typing import Any, Optional

import torch
from diffusers import WanImageToVideoPipeline  # type: ignore[import]

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

BASE_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
LORA_REPO = "lopi999/Wan2.2-I2V_General-NSFW-LoRA"

# LoRA weight filenames
LORA_HIGH = "NSFW-22-H-e8.safetensors"
LORA_LOW = "NSFW-22-L-e8.safetensors"


class ModelVariant(StrEnum):
    """Available Wan 2.2 I2V NSFW LoRA variants."""

    WAN22_I2V_NSFW_HIGH = "2.2_I2V_NSFW_High"
    WAN22_I2V_NSFW_LOW = "2.2_I2V_NSFW_Low"


_LORA_FILES = {
    ModelVariant.WAN22_I2V_NSFW_HIGH: LORA_HIGH,
    ModelVariant.WAN22_I2V_NSFW_LOW: LORA_LOW,
}


class ModelLoader(ForgeModel):
    """Wan 2.2 I2V NSFW LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_NSFW_HIGH: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.WAN22_I2V_NSFW_LOW: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_NSFW_HIGH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_I2V_NSFW_LORA",
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
        """Load the Wan 2.2 I2V pipeline with NSFW LoRA weights applied.

        Returns:
            torch.nn.Module: The Wan transformer model with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        """Prepare synthetic tensor inputs for the Wan transformer.

        Returns:
            dict with hidden_states, timestep, and encoder_hidden_states.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.pipeline.transformer.config

        num_frames = 1
        height = 32
        width = 32
        in_channels = config.in_channels

        hidden_states = torch.randn(
            batch_size, in_channels, num_frames, height, width, dtype=dtype
        )
        timestep = torch.tensor([1], dtype=torch.long).expand(batch_size)
        encoder_hidden_states = torch.randn(
            batch_size, 64, config.text_dim, dtype=dtype
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
