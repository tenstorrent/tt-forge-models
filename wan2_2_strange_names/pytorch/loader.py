# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokosha01/Wan2.2_StrangeNames model loader implementation.

Z-Image-Turbo pipeline variants from Kokosha01/Wan2.2_StrangeNames.
The variant safetensors files contain CLIP/SigLIP vision encoder weights
(not LoRA adapters); all variants return the same base Z-Image-Turbo transformer.

Repository: https://huggingface.co/Kokosha01/Wan2.2_StrangeNames
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline

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

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "Kokosha01/Wan2.2_StrangeNames"


class ModelVariant(StrEnum):
    """Available Kokosha01/Wan2.2_StrangeNames variants."""

    GLASS_ROOT_D2 = "GlassRoot_D2"
    NOVA_MIND_X1 = "NovaMind_X1"
    FROST_BYTE_K7 = "FrostByte_K7"
    IRON_SIGHT_V7 = "IronSight_V7"
    SOLAR_FLINT_L2 = "SolarFlint_L2"
    VELVET_RUSH_Q4 = "VelvetRush_Q4"
    PHANTOM_WEAVE_R5 = "PhantomWeave_R5"
    ECHO_VAULT_T9 = "EchoVault_T9"


class ModelLoader(ForgeModel):
    """Kokosha01/Wan2.2_StrangeNames model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=LORA_REPO)
        for variant in ModelVariant
    }

    DEFAULT_VARIANT = ModelVariant.GLASS_ROOT_D2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wan2.2_StrangeNames",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image-Turbo base pipeline.

        The variant safetensors files in Kokosha01/Wan2.2_StrangeNames contain
        CLIP/SigLIP vision encoder weights (vision_model.* keys), not LoRA adapters.
        The ZImagePipeline has no vision encoder component, so these files cannot
        be applied via load_lora_weights. The base transformer is returned for all
        variants.
        """
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Z-Image-Turbo DiT transformer.

        Returns:
            torch.nn.Module: The Z-Image-Turbo DiT transformer.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A high quality photograph of a mountain landscape"

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
