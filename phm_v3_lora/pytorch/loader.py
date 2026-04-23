# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHM v3 LoRA (Shero448/PHM_v3_lora) model loader implementation.

PHM v3 LoRA is a LoRA adapter for SDXL that generates anime-style illustrations.
It is applied on top of the WAI NSFW Illustrious SDXL v140 base model.

Available variants:
- PHM_V3_LORA: Shero448/PHM_v3_lora LoRA text-to-image generation
"""

from typing import Optional

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


class ModelVariant(StrEnum):
    """Available PHM v3 LoRA model variants."""

    PHM_V3_LORA = "PHM_v3_lora"


class ModelLoader(ForgeModel):
    """PHM v3 LoRA SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PHM_V3_LORA: ModelConfig(
            pretrained_model_name="Shero448/PHM_v3_lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PHM_V3_LORA

    # Base SDXL checkpoint that the LoRA is applied to.
    BASE_MODEL = "dhead/wai-nsfw-illustrious-sdxl-v140-sdxl"
    LORA_FILENAME = "PHM_style_IL_v3.3.safetensors"
    LORA_SCALE = 1.0

    prompt = (
        "(masterpiece, best quality:1.2), amazing quality, very aesthetic, 32k, "
        "absurdres, newest, scenery"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PHM_v3_lora",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the SDXL pipeline with PHM v3 LoRA weights applied.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model with LoRA weights fused.
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

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PHM v3 LoRA UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
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
