#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 I2V Lightning model loader implementation.

Loads the magespace/Wan2.2-I2V-A14B-Lightning-Diffusers pipeline, a distilled
variant of Wan 2.2 I2V optimized for fast inference with fewer denoising steps.
The transformer (WanTransformer3DModel) is extracted from the pipeline and
returned as the testable nn.Module.

Available variants:
- WAN22_I2V_A14B_LIGHTNING: Wan 2.2 Image-to-Video A14B Lightning
"""

from typing import Any, Dict, Optional

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # type: ignore[import]

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

REPO_ID = "magespace/Wan2.2-I2V-A14B-Lightning-Diffusers"

TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 I2V Lightning variants."""

    WAN22_I2V_A14B_LIGHTNING = "2.2_I2V_A14B_Lightning"


class ModelLoader(ForgeModel):
    """Wan 2.2 I2V Lightning model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_I2V_A14B_LIGHTNING: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_I2V_A14B_LIGHTNING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[WanImageToVideoPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_LIGHTNING",
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
        """Load the Wan 2.2 I2V Lightning pipeline and return the transformer.

        Returns:
            WanTransformer3DModel extracted from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        vae = AutoencoderKLWan.from_pretrained(
            REPO_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            REPO_ID,
            vae=vae,
            torch_dtype=dtype,
        )

        return self.pipeline.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic tensor inputs for the WanTransformer3DModel forward pass.

        Returns:
            dict with tensor inputs suitable for WanTransformer3DModel.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        config = self.pipeline.transformer.config
        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
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
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }
