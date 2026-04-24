# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
highscoregames12018/lora-collection model loader implementation.

Loads the Stable Diffusion XL base pipeline and applies a LoRA adapter from
highscoregames12018/lora-collection for stylized text-to-image generation.

Repository: https://huggingface.co/highscoregames12018/lora-collection
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
from .src.model_utils import load_pipe
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available highscoregames12018/lora-collection model variants."""

    ANIME_ARTISTIC_2 = "Anime_artistic_2"


_LORA_FILES = {
    ModelVariant.ANIME_ARTISTIC_2: "Anime_artistic_2.safetensors",
}


class ModelLoader(ForgeModel):
    """highscoregames12018/lora-collection model loader for SDXL text-to-image generation."""

    _VARIANTS = {
        ModelVariant.ANIME_ARTISTIC_2: ModelConfig(
            pretrained_model_name="highscoregames12018/lora-collection",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANIME_ARTISTIC_2

    # Base SDXL model that the LoRA is applied to
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_SCALE = 1.0

    prompt = "masterpiece, best quality, anime artistic style, 1girl, standing in a cherry blossom garden"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="highscoregames_lora_collection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the SDXL pipeline with LoRA weights fused.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model with LoRA weights fused.
        """
        lora_model_id = self._variant_config.pretrained_model_name
        lora_filename = _LORA_FILES[self._variant]

        self.pipeline = load_pipe(
            base_model_name=self.BASE_MODEL,
            lora_model_id=lora_model_id,
            lora_filename=lora_filename,
            lora_scale=self.LORA_SCALE,
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the UNet model.

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
