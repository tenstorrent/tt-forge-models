# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
T2I-Adapter Canny SDXL model loader implementation
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
    load_t2i_adapter_canny_sdxl_pipe,
    create_canny_conditioning_image,
    t2i_adapter_canny_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available T2I-Adapter Canny SDXL model variants."""

    T2I_ADAPTER_CANNY_SDXL_1_0 = "T2I-Adapter_Canny_SDXL_1.0"


class ModelLoader(ForgeModel):
    """T2I-Adapter Canny SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.T2I_ADAPTER_CANNY_SDXL_1_0: ModelConfig(
            pretrained_model_name="TencentARC/t2i-adapter-canny-sdxl-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.T2I_ADAPTER_CANNY_SDXL_1_0

    prompt = "Mystical fairy in real, magic, 4k picture, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="T2I-Adapter Canny SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the T2I-Adapter Canny SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_t2i_adapter_canny_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the T2I-Adapter Canny SDXL UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
                - down_intrablock_additional_residuals (list of torch.Tensor): Adapter residuals
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        adapter_image = create_canny_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            adapter_state,
        ) = t2i_adapter_canny_sdxl_preprocessing(
            self.pipeline, self.prompt, adapter_image
        )

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_intrablock_additional_residuals": adapter_state,
        }
