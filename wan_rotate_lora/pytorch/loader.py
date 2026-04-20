#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.1 I2V Rotate LoRA model loader implementation.

Loads the Wan 2.1 I2V 14B 480P base pipeline and applies the Rotate LoRA
weights from Remade-AI/Rotate for 360-degree rotation image-to-video
generation.
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

BASE_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
LORA_REPO = "Remade-AI/Rotate"
LORA_WEIGHT = "rotate_20_epochs.safetensors"


class ModelVariant(StrEnum):
    """Available Wan 2.1 I2V Rotate LoRA variants."""

    ROTATE = "Rotate"


class ModelLoader(ForgeModel):
    """Wan 2.1 I2V Rotate LoRA model loader."""

    _VARIANTS = {
        ModelVariant.ROTATE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ROTATE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_ROTATE_LORA",
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
        """Load the Wan 2.1 I2V pipeline with Rotate LoRA weights applied.

        Returns:
            torch.nn.Module: The Wan transformer model with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT,
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
