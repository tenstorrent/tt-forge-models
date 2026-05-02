# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qinglong DetailedEyes Z-Image LoRA model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
bdsqlsz/qinglong_DetailedEyes_Z-Image LoRA adapter for detailed
eye generation in text-to-image diffusion.

Reference: https://huggingface.co/bdsqlsz/qinglong_DetailedEyes_Z-Image
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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


class ModelVariant(StrEnum):
    """Available Qinglong DetailedEyes Z-Image model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Qinglong DetailedEyes Z-Image LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="bdsqlsz/qinglong_DetailedEyes_Z-Image",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    base_model = "Tongyi-MAI/Z-Image-Turbo"
    prompt = "A close-up portrait with highly detailed eyes, photorealistic"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qinglong DetailedEyes Z-Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo transformer with DetailedEyes LoRA applied.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Z-Image-Turbo DiT transformer with LoRA weights.
        """
        dtype = dtype_override or torch.bfloat16
        if self.pipeline is None:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.base_model, torch_dtype=dtype, **kwargs
            )
            self.pipeline.load_lora_weights(
                self._variant_config.pretrained_model_name,
            )
            self.pipeline.fuse_lora()
        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)
        return self.pipeline.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Load and return sample inputs for the transformer.

        Returns:
            list: [latent_input_list, timestep, prompt_embeds]
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128

        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=self.prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        num_channels_latents = self.pipeline.transformer.in_channels
        vae_scale = self.pipeline.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
