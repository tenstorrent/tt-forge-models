# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pornmaster v1 Z-Image-Turbo LoRA (RomixERR/Pornmaster_v1-Z-Images-Turbo) model
loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
RomixERR/Pornmaster_v1-Z-Images-Turbo LoRA adapter weights for text-to-image
generation. The repository ships two training checkpoints that are exposed as
variants.

Available variants:
- V1_000043500: Pornmaster_v1_000043500.safetensors LoRA weights
- V1_000044700: Pornmaster_v1_000044700.safetensors LoRA weights
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline

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

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "RomixERR/Pornmaster_v1-Z-Images-Turbo"


class ModelVariant(StrEnum):
    """Available Pornmaster v1 Z-Image-Turbo LoRA model variants."""

    V1_000043500 = "Pornmaster_v1_000043500"
    V1_000044700 = "Pornmaster_v1_000044700"


_LORA_FILES = {
    ModelVariant.V1_000043500: "Pornmaster_v1_000043500.safetensors",
    ModelVariant.V1_000044700: "Pornmaster_v1_000044700.safetensors",
}


class ModelLoader(ForgeModel):
    """Pornmaster v1 Z-Image-Turbo LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_000043500: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
        ModelVariant.V1_000044700: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_000044700

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe: Optional[ZImagePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pornmaster v1 Z-Image-Turbo LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo pipeline with Pornmaster v1 LoRA weights applied."""
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        lora_file = _LORA_FILES[self._variant]
        self._pipe.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )
        self._pipe.fuse_lora()

        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the DiT transformer with Pornmaster v1 LoRA weights fused.

        Returns:
            torch.nn.Module: The Z-Image-Turbo DiT transformer with LoRA applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "pronmstr. A high quality photograph of a mountain landscape"

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

        # Prepare timestep
        timestep = torch.tensor([0.5], dtype=dtype)

        # The transformer expects x as list of [1, channels, 1, H, W] tensors
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
