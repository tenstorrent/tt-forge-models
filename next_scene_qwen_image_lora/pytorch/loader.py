#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Next-Scene Qwen Image LoRA model loader implementation.

Loads the Qwen-Image-Edit base pipeline and applies cinematic next-scene LoRA
weights from lovis93/next-scene-qwen-image-lora-2509 for frame-to-frame
image generation with natural visual progression.

Available variants:
- NEXT_SCENE_V2: Recommended v2 LoRA (next-scene_lora-v2-3000.safetensors)
- NEXT_SCENE_V1: Legacy v1 LoRA (next-scene_lora_v1-3000.safetensors)
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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "lovis93/next-scene-qwen-image-lora-2509"

# LoRA weight filenames
LORA_V2 = "next-scene_lora-v2-3000.safetensors"
LORA_V1 = "next-scene_lora_v1-3000.safetensors"


class ModelVariant(StrEnum):
    """Available Next-Scene Qwen Image LoRA variants."""

    NEXT_SCENE_V2 = "V2"
    NEXT_SCENE_V1 = "V1"


_LORA_FILES = {
    ModelVariant.NEXT_SCENE_V2: LORA_V2,
    ModelVariant.NEXT_SCENE_V1: LORA_V1,
}


class ModelLoader(ForgeModel):
    """Next-Scene Qwen Image LoRA model loader."""

    _VARIANTS = {
        ModelVariant.NEXT_SCENE_V2: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.NEXT_SCENE_V1: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NEXT_SCENE_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NEXT_SCENE_QWEN_IMAGE_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-Edit pipeline with next-scene LoRA weights applied.

        Returns:
            torch.nn.Module: The transformer model with LoRA weights fused.
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

    def load_inputs(self, **kwargs) -> dict:
        """Prepare synthetic tensor inputs for the transformer forward pass.

        Returns:
            dict of tensors matching the QwenImageTransformer2DModel forward signature.
        """
        config = self.pipeline.transformer.config
        dtype = next(self.pipeline.transformer.parameters()).dtype

        batch_size = 1
        in_channels = config.in_channels
        joint_dim = config.joint_attention_dim
        patch_size = config.patch_size
        vae_scale_factor = self.pipeline.vae_scale_factor

        height = 256
        width = 256
        latent_h = height // vae_scale_factor // patch_size
        latent_w = width // vae_scale_factor // patch_size
        seq_len = latent_h * latent_w

        text_seq_len = 64

        hidden_states = torch.randn(batch_size, seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, text_seq_len, dtype=dtype)
        timestep = torch.tensor([500], dtype=torch.long)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": [(1, latent_h, latent_w)],
            "return_dict": False,
        }
