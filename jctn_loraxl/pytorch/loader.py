# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
JCTN_LORAxl (JCTN/JCTN_LORAxl) model loader implementation.

JCTN_LORAxl is a collection of LoRA adapters for SDXL. This loader uses the
"Super Cereal" (cereal_box_sdxl_v1) LoRA, which generates cereal box-style
imagery with humorous themes on top of the Stable Diffusion XL base 1.0
checkpoint.

Available variants:
- JCTN_LORAXL: JCTN/JCTN_LORAxl LoRA text-to-image generation
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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)
from .src.model_utils import load_pipe


class ModelVariant(StrEnum):
    """Available JCTN_LORAxl model variants."""

    JCTN_LORAXL = "JCTN_LORAxl"


class ModelLoader(ForgeModel):
    """JCTN_LORAxl SDXL LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.JCTN_LORAXL: ModelConfig(
            pretrained_model_name="JCTN/JCTN_LORAxl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JCTN_LORAXL

    # Base SDXL checkpoint that the LoRA is applied to.
    BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    LORA_FILENAME = "cereal_box_sdxl_v1.safetensors"
    LORA_SCALE = 1.0

    prompt = "kitty litter crunch, free surprise inside"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="JCTN_LORAxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the SDXL pipeline with JCTN_LORAxl LoRA weights fused.

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

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the JCTN_LORAxl UNet model.

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
