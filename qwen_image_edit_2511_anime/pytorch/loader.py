# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2511 Anime LoRA model loader.

Loads the Qwen/Qwen-Image-Edit-2511 base diffusion pipeline and applies
the prithivMLmods/Qwen-Image-Edit-2511-Anime LoRA adapter for flat
cel-shaded anime-style image editing.

Available variants:
- QWEN_IMAGE_EDIT_2511_ANIME: Anime LoRA (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

BASE_REPO_ID = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO_ID = "prithivMLmods/Qwen-Image-Edit-2511-Anime"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-2511 Anime model variants."""

    QWEN_IMAGE_EDIT_2511_ANIME = "Anime"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2511 Anime LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_2511_ANIME: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_2511_ANIME

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen_Image_Edit_2511_Anime",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Qwen-Image-Edit-2511 pipeline with Anime LoRA weights.

        Returns:
            torch.nn.Module: The transformer component of the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(LORA_REPO_ID)
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs) -> Any:
        """Load synthetic tensor inputs for the QwenImageTransformer2DModel.

        Returns:
            dict: Keyword arguments matching the transformer's forward signature.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        transformer = self.pipeline.transformer

        # Image dimensions and latent packing for 512x512 input
        height, width = 512, 512
        vae_scale_factor = self.pipeline.vae_scale_factor  # 8
        latent_h = 2 * (height // (vae_scale_factor * 2))  # 64
        latent_w = 2 * (width // (vae_scale_factor * 2))  # 64

        in_channels = transformer.config.in_channels  # 64
        joint_attention_dim = transformer.config.joint_attention_dim  # 3584

        # After packing: (batch, (latent_h//2)*(latent_w//2), in_channels)
        patches_per_image = (latent_h // 2) * (latent_w // 2)  # 1024
        # Concatenate noise latents + condition image latents
        hidden_states = torch.randn(
            batch_size, patches_per_image * 2, in_channels, dtype=dtype
        )

        text_seq_len = 128
        encoder_hidden_states = torch.randn(
            batch_size, text_seq_len, joint_attention_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(
            batch_size, text_seq_len, dtype=torch.bool
        )

        # Timestep in [0, 1] range (pipeline divides raw scheduler timestep by 1000)
        timestep = torch.full((batch_size,), 0.5, dtype=dtype)

        # img_shapes: one tuple per image segment (noise + condition)
        img_shape_h = latent_h // 2  # 32
        img_shape_w = latent_w // 2  # 32
        img_shapes = [
            [(1, img_shape_h, img_shape_w), (1, img_shape_h, img_shape_w)]
        ] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
