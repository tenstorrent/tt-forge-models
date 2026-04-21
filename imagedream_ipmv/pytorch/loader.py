# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ImageDream IPMV Diffusers model loader implementation.

ImageDream is an image-prompt multi-view diffusion model for 3D generation.
Given a single input image (and optional text prompt), the MVDreamPipeline
produces a set of consistent multi-view renderings that can be lifted into a
3D asset.

Repository: https://huggingface.co/ashawkey/imagedream-ipmv-diffusers
Paper: https://arxiv.org/abs/2312.02201
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

REPO_ID = "ashawkey/imagedream-ipmv-diffusers"

# UNet config values taken from the published MultiViewUNetModel config.
LATENT_CHANNELS = 4
LATENT_HEIGHT = 32
LATENT_WIDTH = 32
CONTEXT_DIM = 1024
CAMERA_DIM = 16
# CLIP ViT-L/14 image encoder produces 257 tokens (1 CLS + 16x16 patch tokens).
IP_NUM_TOKENS = 257
# ImageDream uses num_frames=4 plus one extra view for the conditioning image.
NUM_FRAMES = 4
ACTUAL_NUM_FRAMES = NUM_FRAMES + 1
# Classifier-free guidance doubles the batch.
CFG_MULTIPLIER = 2


class ModelVariant(StrEnum):
    """Available ImageDream IPMV variants."""

    IPMV = "ipmv"


class ModelLoader(ForgeModel):
    """ImageDream IPMV multi-view UNet loader.

    Loads the full MVDreamPipeline (a custom diffusers pipeline bundled with
    the HuggingFace repository) and returns the MultiViewUNetModel UNet. The
    UNet's forward pass takes keyword arguments describing latents, timesteps,
    text context, multi-view camera parameters, and optional image-prompt
    conditioning.
    """

    _VARIANTS = {
        ModelVariant.IPMV: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.IPMV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ImageDream IPMV",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> DiffusionPipeline:
        if self.pipeline is None:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                custom_pipeline=self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        return self.pipeline

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        return self._load_pipeline(dtype).unet

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch = CFG_MULTIPLIER * ACTUAL_NUM_FRAMES

        sample = torch.randn(
            batch, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
        )
        timesteps = torch.zeros(batch, dtype=dtype)
        context = torch.randn(batch, 77, CONTEXT_DIM, dtype=dtype)
        camera = torch.randn(batch, CAMERA_DIM, dtype=dtype)
        ip = torch.randn(batch, IP_NUM_TOKENS, CONTEXT_DIM, dtype=dtype)
        # ip_img is not repeated per-frame: one latent per CFG branch.
        ip_img = torch.randn(
            CFG_MULTIPLIER, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
        )

        return {
            "x": sample,
            "timesteps": timesteps,
            "context": context,
            "num_frames": ACTUAL_NUM_FRAMES,
            "camera": camera,
            "ip": ip,
            "ip_img": ip_img,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
