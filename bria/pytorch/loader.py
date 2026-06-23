# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BRIA-2.3 text-to-image model loader implementation.

NOTE (bringup status): This loader is an UNTESTED scaffold. briaai/BRIA-2.3 is a
gated HuggingFace repository and the bringup CI token does not have an access
grant, so the weights could not be downloaded, the loader could not be run on
CPU or Tenstorrent hardware, and the architecture details below could not be
confirmed against the real checkpoint. BRIA-2.3 is documented on its public
model card as sharing the Stable Diffusion XL architecture
(StableDiffusionXLPipeline: two CLIP text encoders, a UNet2DConditionModel
denoiser, and a VAE), so this loader mirrors the existing
stable_diffusion_xl/pytorch loader and reuses its SDXL preprocessing helper.
Once access to briaai/BRIA-2.3 is granted, validate (and adjust if needed) the
variant config, preprocessing, and dtype handling before relying on this loader.
"""

import torch
from typing import Optional

from diffusers import DiffusionPipeline

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

# BRIA-2.3 is SDXL-architecture, so reuse the SDXL preprocessing pipeline.
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available BRIA model variants."""

    BRIA_2_3 = "2.3"


class ModelLoader(ForgeModel):
    """BRIA-2.3 text-to-image (SDXL-architecture) model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BRIA_2_3: ModelConfig(
            pretrained_model_name="briaai/BRIA-2.3",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BRIA_2_3

    # Shared configuration parameters
    prompt = "A photorealistic portrait of an astronaut riding a horse on Mars"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BRIA-2.3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BRIA-2.3 pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses float32.

        Returns:
            DiffusionPipeline: The BRIA-2.3 (SDXL-architecture) pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # BRIA-2.3 is gated on HuggingFace; loading requires an access-granted token.
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=torch.float32
        )
        pipe.to("cpu")

        # Put all sub-modules in eval mode and freeze parameters.
        modules = [pipe.text_encoder, pipe.text_encoder_2, pipe.unet, pipe.vae]
        for module in modules:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        self.pipeline = pipe

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BRIA-2.3 denoiser (UNet).

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List: Input tensors that can be fed to the UNet denoiser:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional SDXL conditioning inputs
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
