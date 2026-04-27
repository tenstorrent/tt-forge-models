# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Add Detail XL LoRA model loader implementation.

Add Detail XL is a LoRA adapter for Stable Diffusion XL that modulates the
level of detail in generated images. Positive weights add detail, negative
weights reduce detail. It is applied on top of an SDXL base pipeline.
"""

import warnings

import torch
import torch.nn as nn
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
from .src.model_utils import load_pipe
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class _UNetWrapper(nn.Module):
    """Wraps UNet2DConditionModel to accept flat tensor args instead of added_cond_kwargs dict.

    The TT XLA backend lowers inputs to StableHLO, which only supports tensors.
    Passing added_cond_kwargs as a Python dict causes the compiler to hang. This
    wrapper receives text_embeds and time_ids as separate tensors, builds the dict
    internally, and returns the raw sample tensor from the UNet output.
    """

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample


class ModelVariant(StrEnum):
    """Available Add Detail XL LoRA model variants."""

    ADD_DETAIL_XL = "add-detail-xl"


class ModelLoader(ForgeModel):
    """Add Detail XL LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.ADD_DETAIL_XL: ModelConfig(
            pretrained_model_name="LyliaEngine/add-detail-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ADD_DETAIL_XL

    # Base SDXL model that the LoRA is applied to
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_FILENAME = "add-detail-xl.safetensors"
    LORA_SCALE = 1.5

    prompt = "score_9, score_8_up, score_7_up, a beautiful landscape with mountains and a lake, rating_safe"

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
            model="Add Detail XL LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return a wrapped UNet for the SDXL pipeline with Add Detail XL LoRA.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            _UNetWrapper: Wrapped UNet that accepts flat tensor inputs.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(
            base_model_name=self.BASE_MODEL,
            lora_model_id=pretrained_model_name,
            lora_filename=self.LORA_FILENAME,
            lora_scale=self.LORA_SCALE,
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return _UNetWrapper(self.pipeline.unet)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Flat tensor keyword arguments matching _UNetWrapper.forward signature.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        # retrieve_timesteps → scheduler.set_timesteps also calls np.array(tensor)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="__array__ implementation",
                category=DeprecationWarning,
            )
            (
                latent_model_input,
                timesteps,
                prompt_embeds,
                timestep_cond,
                added_cond_kwargs,
                add_time_ids,
            ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        timestep = timesteps[0]
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            text_embeds = text_embeds.to(dtype_override)
            time_ids = time_ids.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
