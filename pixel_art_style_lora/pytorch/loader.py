#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pixel Art Style LoRA model loader implementation.

Loads the Z-Image-Turbo base pipeline and applies Pixel Art Style LoRA weights
from tarn59/pixel_art_style_lora_z_image_turbo for stylized text-to-image generation.

Available variants:
- BASE: Default pixel art style LoRA on Z-Image-Turbo
"""

from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image  # type: ignore[import]
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
LORA_REPO = "tarn59/pixel_art_style_lora_z_image_turbo"
LORA_FILENAME = "pixel_art_style_z_image_turbo.safetensors"


class ModelVariant(StrEnum):
    """Available Pixel Art Style LoRA variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Pixel Art Style LoRA model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PIXEL_ART_STYLE_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo pipeline with Pixel Art Style LoRA weights applied.

        Returns:
            torch.nn.Module: The transformer component with LoRA weights fused.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        # Load LoRA state dict and inject default alpha values for missing keys
        # (diffusers 0.37+ Z-Image conversion requires alpha entries)
        lora_path = hf_hub_download(LORA_REPO, LORA_FILENAME)
        state_dict = load_file(lora_path)
        for key in list(state_dict.keys()):
            if key.endswith(".lora_A.weight"):
                alpha_key = key.replace(".lora_A.weight", ".alpha")
                if alpha_key not in state_dict:
                    rank = state_dict[key].shape[0]
                    state_dict[alpha_key] = torch.tensor(float(rank))

        self.pipeline.load_lora_weights(state_dict)
        self.pipeline.fuse_lora()

        return self.pipeline.transformer

    def load_inputs(self, batch_size=1, dtype_override=None, **kwargs):
        """Prepare preprocessed tensor inputs for the Z-Image transformer.

        Returns:
            list: [latent_list, timestep, prompt_embeds_list]
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.transformer.dtype
        prompt = "Pixel art style. A small cottage in a forest clearing with warm light glowing from the windows."

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=[prompt] * batch_size,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        in_channels = self.pipeline.transformer.in_channels
        vae_scale = self.pipeline.vae_scale_factor
        height = 1024
        width = 1024
        latent_h = height // vae_scale
        latent_w = width // vae_scale

        latents = torch.randn(batch_size, in_channels, latent_h, latent_w, dtype=dtype)
        latents = latents.unsqueeze(2)
        latent_list = list(latents.unbind(dim=0))

        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

        if dtype_override:
            latent_list = [lat.to(dtype_override) for lat in latent_list]
            timestep = timestep.to(dtype_override)
            prompt_embeds = [pe.to(dtype_override) for pe in prompt_embeds]

        return [latent_list, timestep, prompt_embeds]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack transformer output to a tensor."""
        if isinstance(fwd_output, tuple):
            out = fwd_output[0]
        elif hasattr(fwd_output, "sample"):
            out = fwd_output.sample
        else:
            out = fwd_output
        if isinstance(out, list):
            return torch.stack(out)
        return out
