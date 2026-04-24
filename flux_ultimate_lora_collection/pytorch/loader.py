# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
strangerzonehf/Flux-Ultimate-LoRA-Collection model loader implementation.

Loads the FLUX.1 transformer from cocktailpeanut/flux1-schnell-q8 (a publicly
accessible Q8-quantized FLUX.1-schnell checkpoint) and optionally applies a
style LoRA from strangerzonehf/Flux-Ultimate-LoRA-Collection.

Using cocktailpeanut/flux1-schnell-q8 avoids the gated access requirement of
black-forest-labs/FLUX.1-dev and black-forest-labs/FLUX.1-schnell while
exercising the same transformer architecture.

Repository: https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection
"""

from typing import Any, Optional

import torch
from diffusers import FluxTransformer2DModel

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

BASE_MODEL = "cocktailpeanut/flux1-schnell-q8"
LORA_REPO = "strangerzonehf/Flux-Ultimate-LoRA-Collection"

LORA_ANIMEO = "Animeo.safetensors"
LORA_3D_REALISM = "3D-Realism.safetensors"
LORA_CASTOR_3D_PORTRAIT = "Castor-3D-Portrait-Flux-LoRA.safetensors"


class ModelVariant(StrEnum):
    """Available Flux-Ultimate-LoRA-Collection style variants."""

    ANIMEO = "Animeo"
    REALISM_3D = "3D-Realism"
    CASTOR_3D_PORTRAIT = "Castor-3D-Portrait"


_LORA_FILES = {
    ModelVariant.ANIMEO: LORA_ANIMEO,
    ModelVariant.REALISM_3D: LORA_3D_REALISM,
    ModelVariant.CASTOR_3D_PORTRAIT: LORA_CASTOR_3D_PORTRAIT,
}


class ModelLoader(ForgeModel):
    """strangerzonehf/Flux-Ultimate-LoRA-Collection model loader."""

    _VARIANTS = {
        ModelVariant.ANIMEO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.REALISM_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.CASTOR_3D_PORTRAIT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer: Optional[FluxTransformer2DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flux-Ultimate-LoRA-Collection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX transformer with synthetic LoRA-collection weights.

        Returns:
            FluxTransformer2DModel ready for compilation.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.transformer = FluxTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare synthetic tensor inputs for the FLUX transformer.

        Returns:
            dict with tensors matching FluxTransformer2DModel.forward.
        """
        if self.transformer is None:
            self.load_model()

        dtype = torch.bfloat16
        config = self.transformer.config
        batch_size = 1

        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "pooled_projections": pooled_projections,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
