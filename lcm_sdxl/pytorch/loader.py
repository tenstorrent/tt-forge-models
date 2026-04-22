# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCM-SDXL model loader implementation.

LCM-SDXL (Latent Consistency Model SDXL) is a distilled version of Stable
Diffusion XL that replaces the base UNet with a consistency-trained UNet,
reducing inference from 20-50 steps down to 2-8 steps while maintaining
image quality.

Available variants:
- LCM_SDXL: latent-consistency/lcm-sdxl text-to-image generation
"""

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
from .src.model_utils import load_pipe, lcm_sdxl_preprocessing


class ModelVariant(StrEnum):
    """Available LCM-SDXL model variants."""

    LCM_SDXL = "LCM_SDXL"


class ModelLoader(ForgeModel):
    """LCM-SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.LCM_SDXL: ModelConfig(
            pretrained_model_name="latent-consistency/lcm-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LCM_SDXL

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LCM-SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the LCM-SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The LCM-distilled UNet used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the LCM-SDXL UNet.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
                - timestep_cond (torch.Tensor, optional): Guidance scale embedding
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
        ) = lcm_sdxl_preprocessing(self.pipeline, self.prompt)

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            added_cond_kwargs = {
                k: v.to(dtype_override) if hasattr(v, "to") else v
                for k, v in added_cond_kwargs.items()
            }
            if timestep_cond is not None:
                timestep_cond = timestep_cond.to(dtype_override)

        inputs = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
        if timestep_cond is not None:
            inputs["timestep_cond"] = timestep_cond

        return inputs
