# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance Hyper-SD model loader implementation

Hyper-SD provides accelerated diffusion model inference via distilled LoRA
adapters. This loader applies Hyper-SD LoRA weights to the SDXL base model
for fast text-to-image generation with reduced inference steps.
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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl

LORA_REPO = "ByteDance/Hyper-SD"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class ModelVariant(StrEnum):
    """Available Hyper-SD model variants."""

    SDXL_2STEP_LORA = "SDXL_2step_LoRA"
    SDXL_4STEP_LORA = "SDXL_4step_LoRA"
    SDXL_8STEP_LORA = "SDXL_8step_LoRA"


_LORA_FILES = {
    ModelVariant.SDXL_2STEP_LORA: "Hyper-SDXL-2steps-lora.safetensors",
    ModelVariant.SDXL_4STEP_LORA: "Hyper-SDXL-4steps-lora.safetensors",
    ModelVariant.SDXL_8STEP_LORA: "Hyper-SDXL-8steps-lora.safetensors",
}

_NUM_INFERENCE_STEPS = {
    ModelVariant.SDXL_2STEP_LORA: 2,
    ModelVariant.SDXL_4STEP_LORA: 4,
    ModelVariant.SDXL_8STEP_LORA: 8,
}


class ModelLoader(ForgeModel):
    """ByteDance Hyper-SD model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_2STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_4STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_8STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_2STEP_LORA

    prompt = "a photo of an astronaut riding a horse on mars"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Hyper-SD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SDXL pipeline with Hyper-SD LoRA weights and return the UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        lora_file = _LORA_FILES[self._variant]
        self.pipeline = load_pipe(BASE_MODEL, LORA_REPO, lora_file)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the Hyper-SD UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size for the prompts.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        num_steps = _NUM_INFERENCE_STEPS[self._variant]

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(
            self.pipeline, self.prompt, num_inference_steps=num_steps
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
        }
