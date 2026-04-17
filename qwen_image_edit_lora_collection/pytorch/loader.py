# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit LoRA Collection model loader implementation.

Loads the Qwen/Qwen-Image-Edit-2511 base pipeline and applies LoRA adapters
from strangerzonehf/Qwen-Image-Edit-LoRA-Collection for specialized image
editing tasks. Returns the transformer component for compilation.

Available variants:
- OBJECT_REMOVER_BBOX: Remove objects via bounding box prompts
- GUIDED_HEAD_FACE_SWAP: Guided head/face swapping
- BW_TO_TRUE_COLOR: Convert black-and-white images to true color
- ANIME: Anime style transfer
- UNBLUR_ANYTHING: Unblur/sharpen images
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "strangerzonehf/Qwen-Image-Edit-LoRA-Collection"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit LoRA Collection variants."""

    OBJECT_REMOVER_BBOX = "ObjectRemoverBbox"
    GUIDED_HEAD_FACE_SWAP = "GuidedHeadFaceSwap"
    BW_TO_TRUE_COLOR = "BW2TrueColor"
    ANIME = "Anime"
    UNBLUR_ANYTHING = "UnblurAnything"


_LORA_FILES = {
    ModelVariant.OBJECT_REMOVER_BBOX: "QIE-2511-Object-Remover-Bbox-5000.safetensors",
    ModelVariant.GUIDED_HEAD_FACE_SWAP: "QIE-2511-Guided-Head-Face-Swap-3000.safetensors",
    ModelVariant.BW_TO_TRUE_COLOR: "QIE-2511-BW2TrueColor-3000.safetensors",
    ModelVariant.ANIME: "Qwen-Image-Edit-2511-Anime-2000.safetensors",
    ModelVariant.UNBLUR_ANYTHING: "Qwen-Image-Edit-2511-Unblur-Anything.safetensors",
}

_PROMPTS = {
    ModelVariant.OBJECT_REMOVER_BBOX: "Remove the object in the bounding box",
    ModelVariant.GUIDED_HEAD_FACE_SWAP: "Swap the face with the reference",
    ModelVariant.BW_TO_TRUE_COLOR: "Convert this black and white image to true color",
    ModelVariant.ANIME: "Convert this image to anime style",
    ModelVariant.UNBLUR_ANYTHING: "Sharpen and unblur this image",
}


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit LoRA Collection model loader."""

    _VARIANTS = {
        ModelVariant.OBJECT_REMOVER_BBOX: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.GUIDED_HEAD_FACE_SWAP: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.BW_TO_TRUE_COLOR: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.ANIME: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.UNBLUR_ANYTHING: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OBJECT_REMOVER_BBOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_LORA_COLLECTION",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            **pipe_kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        prompt = _PROMPTS[self._variant]
        height = 128
        width = 128

        image = Image.new("RGB", (256, 256), color=(128, 128, 200))
        condition_images = [self.pipeline.image_processor.resize(image, 384, 384)]

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            image=condition_images,
            prompt=prompt,
            num_images_per_prompt=1,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype)

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        vae_scale_factor = self.pipeline.vae_scale_factor

        h_latent = 2 * (int(height) // (vae_scale_factor * 2))
        w_latent = 2 * (int(width) // (vae_scale_factor * 2))

        latent_shape = (batch_size, 1, num_channels_latents, h_latent, w_latent)
        latents = torch.randn(latent_shape, dtype=dtype)
        latents = self.pipeline._pack_latents(
            latents, batch_size, num_channels_latents, h_latent, w_latent
        )

        img_h = h_latent
        img_w = w_latent
        img_latent_shape = (batch_size, 1, num_channels_latents, img_h, img_w)
        image_latents = torch.randn(img_latent_shape, dtype=dtype)
        image_latents = self.pipeline._pack_latents(
            image_latents, batch_size, num_channels_latents, img_h, img_w
        )

        latent_model_input = torch.cat([latents, image_latents], dim=1)

        img_shapes = [
            [
                (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
                (1, img_h // 2, img_w // 2),
            ]
        ] * batch_size

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
