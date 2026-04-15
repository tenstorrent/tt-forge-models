# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL-Lightning (ByteDance/SDXL-Lightning) model loader implementation.

SDXL-Lightning is a lightning-fast text-to-image generation model distilled from
Stable Diffusion XL using progressive adversarial diffusion distillation.
It can generate high-quality 1024px images in a few steps.

Available variants:
- SDXL_LIGHTNING_4STEP: ByteDance/SDXL-Lightning 4-step UNet variant
"""

import os
from typing import Optional

import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REPO_ID = "ByteDance/SDXL-Lightning"


class ModelVariant(StrEnum):
    """Available SDXL-Lightning model variants."""

    SDXL_LIGHTNING_4STEP = "SDXL_Lightning_4step"


class ModelLoader(ForgeModel):
    """SDXL-Lightning model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_LIGHTNING_4STEP: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SDXL_LIGHTNING_4STEP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL_Lightning",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL-Lightning UNet model.

        Loads the base SDXL UNet and replaces weights with the
        SDXL-Lightning 4-step distilled checkpoint.

        Returns:
            UNet2DConditionModel: The SDXL-Lightning UNet model.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(dtype)
        if not os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            unet.load_state_dict(load_file(hf_hub_download(REPO_ID, ckpt)))

        self.in_channels = unet.config.in_channels
        self.cross_attention_dim = unet.config.cross_attention_dim
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample tensor inputs for the SDXL-Lightning UNet.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder hidden states.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        height, width = 1024, 1024
        seq_len = 77
        # SDXL cross_attention_dim=2048 (CLIP ViT-L 768 + CLIP ViT-bigG 1280)
        encoder_hidden_dim = self.cross_attention_dim
        # CLIP ViT-bigG projection dim for pooled embeddings
        pooled_dim = 1280

        latents = torch.randn(
            (batch_size, self.in_channels, height // 8, width // 8), dtype=dtype
        )

        encoder_hidden_states = torch.randn(
            (batch_size, seq_len, encoder_hidden_dim), dtype=dtype
        )

        text_embeds = torch.randn((batch_size, pooled_dim), dtype=dtype)
        add_time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]], dtype=dtype
        ).repeat(batch_size, 1)

        return {
            "sample": latents,
            "timestep": torch.tensor(999),
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": {
                "text_embeds": text_embeds,
                "time_ids": add_time_ids,
            },
        }
