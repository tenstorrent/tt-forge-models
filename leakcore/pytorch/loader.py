# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LEAKCORE (Mantisum/LEAKCORE) model loader implementation.

LEAKCORE is a LoRA adapter for SDXL that generates amateur-style realistic
images with characteristics such as film grain, mirror selfies, and light leaks.
It is applied on top of a LUSTIFY SDXL checkpoint base model.

Available variants:
- LEAKCORE: Mantisum/LEAKCORE LoRA text-to-image generation
"""

from typing import Optional

import torch

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
from .src.model_utils import load_pipe


class ModelVariant(StrEnum):
    """Available LEAKCORE model variants."""

    LEAKCORE = "LEAKCORE"


class ModelLoader(ForgeModel):
    """LEAKCORE SDXL LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.LEAKCORE: ModelConfig(
            pretrained_model_name="Mantisum/LEAKCORE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LEAKCORE

    # Base SDXL checkpoint that the LoRA is applied to.
    BASE_MODEL = "John6666/lustify-sdxl-nsfw-checkpoint-olt-fixed-textures-sdxl"
    LORA_FILENAME = "leaked_nudes_style_v1_fixed.safetensors"
    LORA_SCALE = 1.0

    prompt = (
        "amateur photo, a woman with long straight blonde hair, taking a mirror selfie, "
        "(film grain:1.0)"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LEAKCORE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SDXL pipeline with LEAKCORE LoRA weights and return the UNet module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet module with LoRA weights fused.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        self.pipeline = load_pipe(
            base_model_name=self.BASE_MODEL,
            lora_model_id=pretrained_model_name,
            lora_filename=self.LORA_FILENAME,
            lora_scale=self.LORA_SCALE,
            dtype=dtype,
        )

        self.pipeline.unet.eval()
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the LEAKCORE UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        timestep = timesteps[0]

        latent_model_input = latent_model_input.to(dtype)
        timestep = timestep.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        added_cond_kwargs = {
            k: v.to(dtype) if isinstance(v, torch.Tensor) else v
            for k, v in added_cond_kwargs.items()
        }

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
