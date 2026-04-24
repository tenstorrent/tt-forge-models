#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 TI2V 5B model loader implementation.

Loads the WanTransformer3DModel from the Wan-AI/Wan2.2-TI2V-5B-Diffusers
companion repository (which provides the model in diffusers format), combined
with scheduler, tokenizer, text encoder, and VAE for the full pipeline.

Wan 2.2 TI2V 5B is a 5B-parameter unified text-to-video (T2V) and
image-to-video (I2V) diffusion transformer that supports 720P@24fps on a
single consumer GPU.

Available variants:
- WAN22_TI2V_5B: Wan 2.2 Text/Image-to-Video 5B
"""

from typing import Any, Optional

import torch

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

BASE_PIPELINE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Small test dimensions for transformer inputs
# Must be divisible by patch_size (1, 2, 2)
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available Wan 2.2 TI2V 5B variants."""

    WAN22_TI2V_5B = "2.2_TI2V_5B"


class ModelLoader(ForgeModel):
    """Wan 2.2 TI2V 5B model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name=BASE_PIPELINE,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_TI2V_5B",
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
        """Load the Wan 2.2 TI2V 5B transformer.

        Loads the transformer from the Wan-AI/Wan2.2-TI2V-5B-Diffusers
        companion repository, which stores the model in diffusers format
        with the correct architecture config for the TI2V 5B variant.
        """
        from diffusers import (
            AutoencoderKLWan,
            WanPipeline,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            transformer = WanTransformer3DModel.from_pretrained(
                BASE_PIPELINE,
                subfolder="transformer",
                torch_dtype=compute_dtype,
            )

            vae = AutoencoderKLWan.from_pretrained(
                BASE_PIPELINE,
                subfolder="vae",
                torch_dtype=torch.float32,
            )

            self.pipeline = WanPipeline.from_pretrained(
                BASE_PIPELINE,
                transformer=transformer,
                vae=vae,
                torch_dtype=compute_dtype,
            )

        return self.pipeline.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare tensor inputs for the Wan 2.2 TI2V 5B transformer forward pass."""
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
