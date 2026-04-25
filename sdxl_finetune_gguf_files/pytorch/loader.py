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
from .src.model_utils import load_gguf_unet, make_sdxl_unet_inputs

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

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._unet = None

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
        """Load and return the SDXL UNet from the GGUF checkpoint.

        Returns:
            UNet2DConditionModel: The loaded UNet instance.
        """
        if self._unet is None:
            self._unet = load_gguf_unet(
                REPO_ID, self.GGUF_FILE, subfolder=self.GGUF_SUBFOLDER
            )

        if dtype_override is not None:
            self._unet = self._unet.to(dtype_override)

        return self._unet

    def load_inputs(self, dtype_override=None):
        """Load and return synthetic inputs for the SDXL UNet.

        Returns:
            list: [latent_model_input, timestep, prompt_embeds, added_cond_kwargs]
        """
        if self._unet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else None
        (
            latent_model_input,
            timestep,
            prompt_embeds,
            added_cond_kwargs,
        ) = make_sdxl_unet_inputs(self._unet, dtype=dtype or self._unet.dtype)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
