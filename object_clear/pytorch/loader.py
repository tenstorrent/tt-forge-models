# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ObjectClear model loader implementation.

ObjectClear is a fine-tuned Stable Diffusion XL inpainting model that performs
complete object removal via an Object-Effect Attention mechanism.
Source: https://huggingface.co/jixin0101/ObjectClear
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
from .src.model_utils import (
    load_object_clear_pipe,
    create_dummy_input_image_and_mask,
    object_clear_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ObjectClear model variants."""

    OBJECT_CLEAR = "object-clear"


class ModelLoader(ForgeModel):
    """ObjectClear model loader implementation."""

    _VARIANTS = {
        ModelVariant.OBJECT_CLEAR: ModelConfig(
            pretrained_model_name="jixin0101/ObjectClear",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OBJECT_CLEAR

    prompt = "a clean background"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ObjectClear",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ObjectClear SDXL inpainting pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_object_clear_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ObjectClear UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Noise latents concatenated with mask and masked image latents
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image, mask_image = create_dummy_input_image_and_mask()

        (
            scaled_latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = object_clear_preprocessing(
            self.pipeline, self.prompt, input_image, mask_image
        )

        timestep = timesteps[0]

        if dtype_override:
            scaled_latent_model_input = scaled_latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": scaled_latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
