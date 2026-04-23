#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 360 Diffusion LoRA model loader implementation.

Loads the Qwen-Image base pipeline and applies the 360-degree equirectangular
panorama LoRA weights from ProGamerGov/qwen-360-diffusion for text-to-image
generation of 360-degree panoramic images.

Available variants:
- INT8_V1: int8-bf16 v1 LoRA on Qwen/Qwen-Image (default)
- INT4_V1: int4-bf16 v1 LoRA on Qwen/Qwen-Image
- INT4_V1B: int4-bf16 v1-b LoRA on Qwen/Qwen-Image
- INT8_V2_2512: int8-bf16 v2 LoRA on Qwen/Qwen-Image-2512
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline  # type: ignore[import]

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

BASE_MODEL_QWEN_IMAGE = "Qwen/Qwen-Image"
BASE_MODEL_QWEN_IMAGE_2512 = "Qwen/Qwen-Image-2512"
LORA_REPO = "ProGamerGov/qwen-360-diffusion"

LORA_INT8_V1 = "qwen-360-diffusion-int8-bf16-v1.safetensors"
LORA_INT4_V1 = "qwen-360-diffusion-int4-bf16-v1.safetensors"
LORA_INT4_V1B = "qwen-360-diffusion-int4-bf16-v1-b.safetensors"
LORA_INT8_V2_2512 = "qwen-360-diffusion-2512-int8-bf16-v2.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen 360 Diffusion LoRA variants."""

    INT8_V1 = "int8_v1"
    INT4_V1 = "int4_v1"
    INT4_V1B = "int4_v1b"
    INT8_V2_2512 = "int8_v2_2512"


_LORA_FILES = {
    ModelVariant.INT8_V1: LORA_INT8_V1,
    ModelVariant.INT4_V1: LORA_INT4_V1,
    ModelVariant.INT4_V1B: LORA_INT4_V1B,
    ModelVariant.INT8_V2_2512: LORA_INT8_V2_2512,
}


class ModelLoader(ForgeModel):
    """Qwen 360 Diffusion LoRA model loader."""

    _VARIANTS = {
        ModelVariant.INT8_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT4_V1B: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE,
        ),
        ModelVariant.INT8_V2_2512: ModelConfig(
            pretrained_model_name=BASE_MODEL_QWEN_IMAGE_2512,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INT8_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_360_DIFFUSION",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image pipeline with 360 diffusion LoRA weights applied.

        Returns:
            QwenImageTransformer2DModel with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        """Prepare tensor inputs for the QwenImageTransformer2DModel.

        Returns:
            dict of input tensors for the transformer.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        config = self.pipeline.transformer.config

        # Use a small image for testing: 64x64 pixels
        height = 64
        width = 64
        vae_scale_factor = 8

        # Latent dimensions after VAE compression (must be divisible by 2 for packing)
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))

        # Packed patch dimensions: 2x2 spatial patches
        num_patches = (latent_h // 2) * (latent_w // 2)
        num_channels_latents = (
            config.in_channels // 4
        )  # in_channels = num_channels * patch_size^2

        # hidden_states: (batch, num_patches, in_channels)
        hidden_states = torch.randn(
            batch_size, num_patches, config.in_channels, dtype=dtype
        )

        # Text embeddings: (batch, text_seq_len, joint_attention_dim)
        text_seq_len = 64
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, config.joint_attention_dim, dtype=dtype
        )

        # Attention mask: all tokens valid
        encoder_hidden_states_mask = torch.ones(
            batch_size, text_seq_len, dtype=torch.bool
        )

        # Timestep (already normalized by 1000 before passing to transformer)
        timestep = torch.tensor([0.5] * batch_size, dtype=dtype)

        # img_shapes: per-sample list of (t, h, w) tuples for RoPE computation
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
