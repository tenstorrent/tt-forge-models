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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)
from .src.model_utils import load_pipe


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
        """Load and return the SDXL pipeline with PHM v3 LoRA weights applied.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionXLPipeline: The pipeline with LoRA weights fused.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(
            base_model_name=self.BASE_MODEL,
            lora_model_id=pretrained_model_name,
            lora_filename=self.LORA_FILENAME,
            lora_scale=self.LORA_SCALE,
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the PHM v3 LoRA model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the UNet model.
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

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
