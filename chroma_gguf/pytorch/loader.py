# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chroma GGUF model loader implementation for text-to-image generation
"""
import os
import torch
from diffusers.models import ChromaTransformer2DModel
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
    """Available Chroma GGUF model variants."""

    UNLOCKED_V50_Q4_0 = "Unlocked_V50_Q4_0"


GGUF_BASE_URL = "https://huggingface.co/silveroxides/Chroma-GGUF/blob/main"
CHROMA_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "flux_config")


class ModelLoader(ForgeModel):
    """Chroma GGUF model loader implementation for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.UNLOCKED_V50_Q4_0: ModelConfig(
            pretrained_model_name="silveroxides/Chroma-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.UNLOCKED_V50_Q4_0: "chroma-unlocked-v50/chroma-unlocked-v50-Q4_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.UNLOCKED_V50_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Chroma GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = self._GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = ChromaTransformer2DModel.from_single_file(
            gguf_url,
            config=CHROMA_CONFIG_DIR,
            **load_kwargs,
        )

        if dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
