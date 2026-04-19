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

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

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

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        )
        prompt = (
            "cinematic polaroid with soft grain subtle vignette gentle lighting "
            "white frame handwritten photographed by prithivMLmods preserving "
            "realistic texture and details"
        )

        height = 512
        width = 512
        vae_scale_factor = self.pipeline.vae_scale_factor

        prompt_embeds, _ = self.pipeline.encode_prompt(prompt=prompt, image=image)

        vae_image = self.pipeline.image_processor.preprocess(
            image, height, width
        ).unsqueeze(2)
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, image_latents = self.pipeline.prepare_latents(
            [vae_image],
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=self.pipeline.transformer.dtype,
            device="cpu",
            generator=torch.Generator().manual_seed(42),
        )

        hidden_states = (
            torch.cat([latents, image_latents], dim=1)
            if image_latents is not None
            else latents
        )
        latent_h = height // vae_scale_factor // 2
        latent_w = width // vae_scale_factor // 2
        img_shapes = [[(1, latent_h, latent_w), (1, latent_h, latent_w)]]
        timestep = torch.tensor([0.5], dtype=self.pipeline.transformer.dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
