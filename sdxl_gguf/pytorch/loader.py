# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL GGUF (hum-ma/SDXL-models-GGUF) model loader implementation.

Stable Diffusion XL is a text-to-image generation model in GGUF quantized format,
based on the SDXL UNet architecture with 3B parameters.

Available variants:
- SDXL_1_0_Q4_0: Q4_0 quantized SDXL 1.0 base model
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
from .src.model_utils import load_sdxl_gguf_pipe, stable_diffusion_preprocessing_xl

REPO_ID = "hum-ma/SDXL-models-GGUF"


class ModelVariant(StrEnum):
    """Available SDXL GGUF model variants."""

    SDXL_1_0_Q4_0 = "sdxl_1.0_Q4_0"


class ModelLoader(ForgeModel):
    """SDXL GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_1_0_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_1_0_Q4_0

    GGUF_FILE = "stable-diffusion-xl-base-1.0-Q4_0.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self.pipeline = load_sdxl_gguf_pipe(REPO_ID, self.GGUF_FILE)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
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
