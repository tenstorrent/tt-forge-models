# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL Finetune GGUF Files (Old-Fisherman/SDXL_Finetune_GGUF_Files) model loader
implementation.

Old-Fisherman hosts a collection of community SDXL finetunes pre-quantized to
GGUF (Q4_K_S / Q5_K_S / Q5_K_M). All variants share the SDXL architecture and
are loaded through diffusers' GGUF quantization support.

Available variants:
- JUGGERNAUT_XL_XI_Q4_K_S: Juggernaut XL XI by RunDiffusion, Q4_K_S quantized
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
from .src.model_utils import load_gguf_pipe, stable_diffusion_preprocessing_xl

REPO_ID = "Old-Fisherman/SDXL_Finetune_GGUF_Files"


class ModelVariant(StrEnum):
    """Available SDXL Finetune GGUF Files model variants."""

    JUGGERNAUT_XL_XI_Q4_K_S = "juggernautXL_juggXIByRundiffusion_Q4_K_S"


class ModelLoader(ForgeModel):
    """SDXL Finetune GGUF Files model loader implementation."""

    _VARIANTS = {
        ModelVariant.JUGGERNAUT_XL_XI_Q4_K_S: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JUGGERNAUT_XL_XI_Q4_K_S

    # GGUF files live under a GGUF_Models/ subfolder in the upstream repo.
    GGUF_SUBFOLDER = "GGUF_Models"
    GGUF_FILE = "juggernautXL_juggXIByRundiffusion_Q4_K_S.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL Finetune GGUF Files",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL finetune pipeline from GGUF checkpoint.

        Returns:
            DiffusionPipeline: The loaded pipeline instance.
        """
        if self.pipeline is None:
            self.pipeline = load_gguf_pipe(
                REPO_ID, self.GGUF_FILE, subfolder=self.GGUF_SUBFOLDER
            )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            list: Input tensors for the UNet model.
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
