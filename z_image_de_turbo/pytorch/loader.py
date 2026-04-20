# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-De-Turbo model loader implementation.

Loads the ostris/Z-Image-De-Turbo de-distilled diffusion transformer and
pairs it with the Tongyi-MAI/Z-Image-Turbo base pipeline for text-to-image
generation.

Available variants:
- Z_IMAGE_DE_TURBO: de-distilled DiT transformer from ostris/Z-Image-De-Turbo
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel

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

REPO_ID = "ostris/Z-Image-De-Turbo"
BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available Z-Image-De-Turbo model variants."""

    Z_IMAGE_DE_TURBO = "Z-Image-De-Turbo"


class ModelLoader(ForgeModel):
    """Z-Image-De-Turbo model loader."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_DE_TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_DE_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe: Optional[ZImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_DE_TURBO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo base pipeline with the de-distilled transformer."""
        transformer = ZImageTransformer2DModel.from_pretrained(
            REPO_ID,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_REPO_ID,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the de-distilled DiT transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare transformer inputs (latents, timestep, prompt_embeds)."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        # Encode the prompt
        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        # Prepare latents
        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        # Prepare timestep (normalized: 0 = fully denoised, 1 = fully noised)
        timestep = torch.tensor([0.5], dtype=dtype)

        # The transformer expects:
        #   x: list of tensors [1, channels, 1, H, W] (one per batch)
        #   t: timestep tensor
        #   cap_feats: prompt_embeds (list of text embeddings)
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
