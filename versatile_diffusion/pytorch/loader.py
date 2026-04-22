# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Versatile Diffusion model loader implementation
"""

import torch
from typing import Optional

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
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.deprecated.versatile_diffusion.modeling_text_unet import (
    UNetFlatConditionModel,
)
from diffusers.pipelines.deprecated.versatile_diffusion.pipeline_versatile_diffusion_text_to_image import (
    VersatileDiffusionTextToImagePipeline,
)
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available Versatile Diffusion model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Versatile Diffusion model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="shi-labs/versatile-diffusion",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Versatile Diffusion",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Versatile Diffusion text-to-image pipeline from Hugging Face.

        Loads each component individually to work around a diffusers compatibility
        issue where newer diffusers versions no longer recognize the 'versatile_diffusion'
        module reference in model_index.json during from_pretrained validation.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            VersatileDiffusionTextToImagePipeline: The pre-trained Versatile Diffusion pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype
        )
        image_unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="image_unet", torch_dtype=dtype
        )
        text_unet = UNetFlatConditionModel.from_pretrained(
            model_name, subfolder="text_unet", torch_dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", torch_dtype=dtype
        )
        scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

        pipe = VersatileDiffusionTextToImagePipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Versatile Diffusion model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "an astronaut riding on a horse on mars",
        ] * batch_size
        return prompt
