# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boltning HyperD SDXL model loader implementation.

Boltning HyperD SDXL is a speed-optimized Stable Diffusion XL checkpoint for
fast text-to-image generation. This loader extracts the UNet component from
the pipeline for compilation.

Available variants:
- BOLTNING_HYPERD_SDXL: GraydientPlatformAPI/boltning-hyperd-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Boltning HyperD SDXL model variants."""

    BOLTNING_HYPERD_SDXL = "boltning-hyperd-sdxl"


class ModelLoader(ForgeModel):
    """Boltning HyperD SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.BOLTNING_HYPERD_SDXL: ModelConfig(
            pretrained_model_name="GraydientPlatformAPI/boltning-hyperd-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BOLTNING_HYPERD_SDXL

    prompt = "A cinematic photo of a lighthouse on a cliff during a storm, dramatic lighting, high detail"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Boltning_HyperD_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Boltning HyperD SDXL pipeline and return the UNet component.

        Returns:
            UNet2DConditionModel: The UNet component from the SDXL pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.to("cpu")

        modules = [
            self.pipeline.text_encoder,
            self.pipeline.unet,
            self.pipeline.text_encoder_2,
            self.pipeline.vae,
        ]
        for module in modules:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return preprocessed inputs for the UNet model.

        Returns:
            list: Preprocessed input tensors for the UNet.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
