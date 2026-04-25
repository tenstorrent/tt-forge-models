# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 White Film to Rendering LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-White_film_to_rendering LoRA adapter on
top of the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for converting
white-model (white film) renders into textured/material renderings.
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
    """Available Qwen Image Edit 2509 White Film to Rendering model variants."""

    WHITE_FILM_TO_RENDERING = "White_Film_To_Rendering"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 White Film to Rendering LoRA model loader."""

    _VARIANTS = {
        ModelVariant.WHITE_FILM_TO_RENDERING: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Image-Edit-2509-White_film_to_rendering",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHITE_FILM_TO_RENDERING

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.WHITE_FILM_TO_RENDERING: "白膜转材质.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPlusPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 White Film to Rendering",
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
        dtype = dtype_override or torch.bfloat16

        # Transformer config: in_channels=64, joint_attention_dim=3584
        # VAE scale factor = 2**3 = 8 (temperal_downsample has 3 entries)
        img_h, img_w = 512, 512
        vae_scale_factor = 8
        # Latent spatial dims after VAE encoding, adjusted for 2x patch packing
        latent_h = 2 * (img_h // (vae_scale_factor * 2))  # 64
        latent_w = 2 * (img_w // (vae_scale_factor * 2))  # 64
        in_channels = 64  # transformer.config.in_channels
        joint_attention_dim = 3584  # transformer.config.joint_attention_dim

        # Packed latent sequence length: (latent_h//2) * (latent_w//2) = 1024
        latent_seq = (latent_h // 2) * (latent_w // 2)

        # Concatenate noise latents and condition image latents along sequence dim
        hidden_states = torch.zeros(
            batch_size, 2 * latent_seq, in_channels, dtype=dtype
        )

        text_seq_len = 32
        encoder_hidden_states = torch.zeros(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(
            batch_size, text_seq_len, dtype=torch.bool
        )

        # Timestep normalized to [0, 1] range (pipeline divides by 1000 before calling transformer)
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

        # RoPE image shapes: one tuple per image slot (generated + condition)
        # Each tuple is (temporal_frames, spatial_h, spatial_w) in patch units
        patch_h = img_h // vae_scale_factor // 2  # 32
        patch_w = img_w // vae_scale_factor // 2  # 32
        img_shapes = [[(1, patch_h, patch_w), (1, patch_h, patch_w)]] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
