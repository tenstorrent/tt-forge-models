# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX-SRPO GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX-SRPO model published
in MonsterMMORPG/Wan_GGUF. FLUX-SRPO is a Stepwise Relative Preference
Optimization fine-tune of FLUX.1-dev. The GGUF transformer is loaded via
diffusers' FluxTransformer2DModel.from_single_file using a local bundled
config to avoid dependency on the gated black-forest-labs/FLUX.1-dev repo.
"""

import importlib.metadata
import os
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

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

GGUF_REPO = "MonsterMMORPG/Wan_GGUF"

_LOCAL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


class ModelVariant(StrEnum):
    """Available FLUX-SRPO GGUF quantization variants."""

    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K: "FLUX-SRPO-GGUF_Q4_K.gguf",
    ModelVariant.Q5_K: "FLUX-SRPO-GGUF_Q5_K.gguf",
    ModelVariant.Q6_K: "FLUX-SRPO-GGUF_Q6_K.gguf",
    ModelVariant.Q8_0: "FLUX-SRPO-GGUF_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX-SRPO GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX-SRPO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_gguf_availability():
        """Patch diffusers _gguf_available flag when gguf is installed at runtime.

        diffusers sets _gguf_available at import time. When gguf is installed later
        by RequirementsManager, the cached flag stays False. Patch it directly.
        """
        import diffusers.utils.import_utils as _import_utils

        if not _import_utils.is_gguf_available():
            try:
                importlib.metadata.version("gguf")
                _import_utils._gguf_available = True
            except importlib.metadata.PackageNotFoundError:
                pass

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX-SRPO transformer.

        Returns:
            torch.nn.Module: The FLUX-SRPO transformer model instance.
        """
        self._fix_gguf_availability()
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        gguf_file = _GGUF_FILES[self._variant]

        self.transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=_LOCAL_CONFIG_DIR,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic inputs for the FLUX-SRPO transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

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

        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)
        guidance = torch.tensor([3.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
