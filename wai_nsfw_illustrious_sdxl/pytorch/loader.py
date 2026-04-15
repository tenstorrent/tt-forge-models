# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAI NSFW Illustrious SDXL v140 (dhead/wai-nsfw-illustrious-sdxl-v140-sdxl) model loader implementation.

WAI NSFW Illustrious SDXL is a Stable Diffusion XL model fine-tuned for
photorealistic text-to-image generation.

Available variants:
- WAI_NSFW_ILLUSTRIOUS_SDXL_V140: dhead/wai-nsfw-illustrious-sdxl-v140-sdxl text-to-image generation
"""

import os
from typing import Optional

import torch
from diffusers import UNet2DConditionModel

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


REPO_ID = "dhead/wai-nsfw-illustrious-sdxl-v140-sdxl"


class ModelVariant(StrEnum):
    """Available WAI NSFW Illustrious SDXL model variants."""

    WAI_NSFW_ILLUSTRIOUS_SDXL_V140 = "WAI_NSFW_Illustrious_SDXL_v140"


class ModelLoader(ForgeModel):
    """WAI NSFW Illustrious SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAI_NSFW_ILLUSTRIOUS_SDXL_V140: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAI_NSFW_ILLUSTRIOUS_SDXL_V140

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAI_NSFW_Illustrious_SDXL_v140",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WAI NSFW Illustrious SDXL UNet model.

        Returns:
            UNet2DConditionModel: The UNet component from the SDXL pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        repo = self._variant_config.pretrained_model_name

        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            unet = UNet2DConditionModel.from_config(repo, subfolder="unet").to(dtype)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                repo, subfolder="unet", torch_dtype=dtype
            )

        self.in_channels = unet.config.in_channels
        self.cross_attention_dim = unet.config.cross_attention_dim
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample tensor inputs for the WAI NSFW Illustrious SDXL UNet.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder hidden states.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        height, width = 1024, 1024
        seq_len = 77
        encoder_hidden_dim = self.cross_attention_dim
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
