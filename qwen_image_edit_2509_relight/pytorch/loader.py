# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 Relight LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-Relight LoRA adapter on top of the
Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for image relighting.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
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


class ModelVariant(StrEnum):
    """Available Qwen Image Edit Relight model variants."""

    RELIGHT = "relight"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 Relight LoRA model loader."""

    _VARIANTS = {
        ModelVariant.RELIGHT: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Image-Edit-2509-Relight",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RELIGHT

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.RELIGHT: "Qwen-Edit-Relight.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 Relight",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.pipeline.transformer.config

        in_channels = config.in_channels
        num_channels_latents = in_channels // 4
        joint_attention_dim = config.joint_attention_dim

        # Synthetic 512x512 image, vae_scale_factor=8
        orig_h, orig_w = 512, 512
        vae_scale_factor = 8
        height = 2 * (orig_h // (vae_scale_factor * 2))
        width = 2 * (orig_w // (vae_scale_factor * 2))

        # Packed latent tokens per image: (height//2) * (width//2)
        latents_seq_len = (height // 2) * (width // 2)

        # Noise latents and conditioning image latents concatenated on sequence dim
        hidden_states = torch.randn(
            batch_size,
            2 * latents_seq_len,
            num_channels_latents * 4,
            dtype=dtype,
        )

        max_seq_len = 256
        encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, joint_attention_dim, dtype=dtype
        )

        timestep = torch.full((batch_size,), 0.5, dtype=dtype)

        latent_grid = orig_h // vae_scale_factor // 2
        img_shapes = [
            [(1, latent_grid, latent_grid), (1, latent_grid, latent_grid)]
        ] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
