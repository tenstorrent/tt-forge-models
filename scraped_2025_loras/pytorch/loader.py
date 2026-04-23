# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup/Scraped_2025_Loras model loader implementation.

Loads a Stable Diffusion XL family base pipeline and applies a LoRA adapter
from MLbackup/Scraped_2025_Loras for text-to-image generation. The repo is a
collection of LoRAs targeting several SDXL-lineage base models (Illustrious XL,
Pony XL, NoobAI XL, and vanilla SDXL); each variant below pairs one LoRA
safetensors file with its matching base model.

Available variants:
- ARCANE_STYLE_ILLUSTRIOUS: Arcane style LoRA on Illustrious XL
- ARCANE_STYLE_PONYXL: Arcane style LoRA on Pony Diffusion V6 XL
- NOOBAI_32K_UHD_AESTHETIC: 32k UHD aesthetic LoRA on NoobAI XL 1.1
- MICRO_CUBE_WORLDS_SDXL: Micro Cube Worlds LoRA on SDXL base 1.0
"""

from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image

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


LORA_REPO = "MLbackup/Scraped_2025_Loras"


class ModelVariant(StrEnum):
    """Available MLbackup Scraped_2025_Loras variants."""

    ARCANE_STYLE_ILLUSTRIOUS = "Arcane_Style_IllustriousXL"
    ARCANE_STYLE_PONYXL = "Arcane_Style_PonyXL"
    NOOBAI_32K_UHD_AESTHETIC = "NoobAI_32k_UHD_Aesthetic"
    MICRO_CUBE_WORLDS_SDXL = "Micro_Cube_Worlds_SDXL_BASE"


_LORA_FILES = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "Arcane_Style_IllustriousXL.safetensors",
    ModelVariant.ARCANE_STYLE_PONYXL: "Arcane_Style_PonyXL.safetensors",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "NoobAI_32k_UHD_Aesthetic.safetensors",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "Micro_Cube_Worlds_SDXL_BASE.safetensors",
}

_BASE_MODELS = {
    ModelVariant.ARCANE_STYLE_ILLUSTRIOUS: "OnomaAIResearch/Illustrious-xl-early-release-v0",
    ModelVariant.ARCANE_STYLE_PONYXL: "AstraliteHeart/pony-diffusion-v6",
    ModelVariant.NOOBAI_32K_UHD_AESTHETIC: "Laxhar/noobai-XL-1.1",
    ModelVariant.MICRO_CUBE_WORLDS_SDXL: "stabilityai/stable-diffusion-xl-base-1.0",
}


class ModelLoader(ForgeModel):
    """MLbackup Scraped_2025_Loras model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=_BASE_MODELS[variant])
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.ARCANE_STYLE_ILLUSTRIOUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Scraped_2025_Loras",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load an SDXL-family pipeline, apply LoRA weights, and return the UNet.

        Returns:
            torch.nn.Module: The UNet model with fused LoRA weights.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )
        self.pipeline.fuse_lora()

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return tensor inputs for the SDXL UNet forward pass.

        Returns:
            dict: Keyword arguments for UNet forward: sample, timestep,
                encoder_hidden_states, added_cond_kwargs.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype
        prompt = "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                device="cpu",
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
            )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

        sample_size = self.pipeline.unet.config.sample_size
        vae_scale = self.pipeline.vae_scale_factor
        height = sample_size * vae_scale
        width = height
        proj_dim = self.pipeline.text_encoder_2.config.projection_dim
        add_time_ids = self.pipeline._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=proj_dim,
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        in_channels = self.pipeline.unet.config.in_channels
        latent_sample = torch.randn(
            2 * batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([999], dtype=dtype)

        if dtype_override is not None:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_sample,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            },
        }

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
