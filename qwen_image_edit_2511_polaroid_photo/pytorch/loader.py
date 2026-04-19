# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2511 Polaroid Photo LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo LoRA adapter for
cinematic Polaroid-style image editing.

Available variants:
- QWEN_IMAGE_EDIT_2511_POLAROID_PHOTO: Polaroid Photo LoRA (bf16)
"""

from typing import Optional

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2511 Polaroid Photo model variants."""

    QWEN_IMAGE_EDIT_2511_POLAROID_PHOTO = "Polaroid_Photo"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2511 Polaroid Photo LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2511_POLAROID_PHOTO: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2511_POLAROID_PHOTO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen_Image_Edit_2511_Polaroid_Photo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(LORA_REPO_ID)
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> dict:
        if self.pipeline is None:
            self.load_model()

        config = self.pipeline.transformer.config
        dtype = self.pipeline.transformer.dtype
        vae_scale_factor = self.pipeline.vae_scale_factor
        height = 512
        width = 512
        latent_h = height // vae_scale_factor // 2
        latent_w = width // vae_scale_factor // 2
        num_channels_latents = config.in_channels // 4
        seq_len = latent_h * latent_w

        hidden_states = torch.randn(1, seq_len * 2, num_channels_latents, dtype=dtype)
        prompt_embeds = torch.randn(1, 64, config.joint_attention_dim, dtype=dtype)
        img_shapes = [[(1, latent_h, latent_w), (1, latent_h, latent_w)]]
        timestep = torch.tensor([0.5], dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
