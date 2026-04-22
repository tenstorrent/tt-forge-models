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

import importlib.util
import os
import sys
from typing import Any, Optional

import torch

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
# image encoder (CLIP ViT-H/14) produces 257 tokens with hidden_size=1280.
IP_NUM_TOKENS = 257
IP_EMBED_DIM = 1280
# ImageDream uses num_frames=4 plus one extra view for the conditioning image.
NUM_FRAMES = 4
ACTUAL_NUM_FRAMES = NUM_FRAMES + 1
# Classifier-free guidance doubles the batch.
CFG_MULTIPLIER = 2


def _load_mv_unet_module():
    """Load the local patched mv_unet module and register it in sys.modules."""
    mv_unet_path = os.path.join(os.path.dirname(__file__), "mv_unet.py")
    spec = importlib.util.spec_from_file_location("mv_unet", mv_unet_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mv_unet"] = module
    spec.loader.exec_module(module)
    return module


class ModelVariant(StrEnum):
    """Available ImageDream IPMV variants."""

    IPMV = "ipmv"


class ModelLoader(ForgeModel):
    """ImageDream IPMV multi-view UNet loader.

    Loads MultiViewUNetModel directly from the HuggingFace repository using a
    locally bundled mv_unet.py (patched to make xformers/kiui optional). The
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
        self._unet = None

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

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        if self._unet is None:
            dtype = dtype_override if dtype_override is not None else torch.float32
            mv_unet = _load_mv_unet_module()
            self._unet = mv_unet.MultiViewUNetModel.from_pretrained(
                self._variant_config.pretrained_model_name,
                subfolder="unet",
                torch_dtype=dtype,
            )
        return self._unet

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
        ip = torch.randn(batch, IP_NUM_TOKENS, IP_EMBED_DIM, dtype=dtype)
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
