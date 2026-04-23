#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 TI2V 5B native-format model loader implementation.

Loads the hybrid text-and-image-to-video transformer from the native
Wan-AI/Wan2.2-TI2V-5B repository (sharded diffusion_pytorch_model safetensors)
and builds a WanPipeline using the Diffusers companion repo for the scheduler,
tokenizer, text encoder, and VAE.

Wan 2.2 TI2V 5B is a 5B-parameter unified text-to-video (T2V) and
image-to-video (I2V) diffusion transformer that supports 720P@24fps on a
single consumer GPU.

Available variants:
- WAN22_TI2V_5B: Wan 2.2 Text/Image-to-Video 5B (native format)
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

NATIVE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
BASE_PIPELINE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan 2.2 TI2V 5B native-format variants."""

    WAN22_TI2V_5B = "2.2_TI2V_5B"


class ModelLoader(ForgeModel):
    """Wan 2.2 TI2V 5B native-format model loader."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B: ModelConfig(
            pretrained_model_name=NATIVE_REPO,
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
        """Load the native-format Wan 2.2 TI2V 5B transformer and build the pipeline.

        Loads the sharded diffusion_pytorch_model safetensors from the native
        Wan-AI/Wan2.2-TI2V-5B repo and combines with the Diffusers companion
        repo's scheduler, tokenizer, text encoder, and VAE (VAE kept in float32
        for numerical stability).
        """
        from diffusers import (
            AutoencoderKLWan,
            WanPipeline,
            WanTransformer3DModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

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

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-video generation."""
        if prompt is None:
            prompt = (
                "Astronaut in a jungle, cold color palette, muted colors, "
                "detailed, 8k"
            )

        return {
            "prompt": prompt,
            "height": 480,
            "width": 832,
            "num_frames": 9,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
        }
