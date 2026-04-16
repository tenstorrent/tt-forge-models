# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Amanatsu Illustrious v1.1 SDXL (John6666/amanatsu-illustrious-v11-sdxl) model loader implementation.

Amanatsu Illustrious is an anime-focused Stable Diffusion XL fine-tune
optimized for detailed anime illustration generation.

Available variants:
- AMANATSU_ILLUSTRIOUS_V11: John6666/amanatsu-illustrious-v11-sdxl text-to-image generation
"""

import os
from typing import Optional

import torch

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "John6666/amanatsu-illustrious-v11-sdxl"

# SDXL UNet input shapes (with classifier-free guidance, batch doubles)
_SDXL_LATENT_CHANNELS = 4
# Use 32x32 latents (256x256 effective image size) for random-weights mode to
# reduce self-attention sequence length from 1024 to 256 for CPU tractability
_SDXL_LATENT_HEIGHT = 32  # 256 / vae_scale_factor(8)
_SDXL_LATENT_WIDTH = 32
_SDXL_SEQ_LEN = 77
_SDXL_ENCODER_HIDDEN_SIZE = 2048  # SDXL dual text encoder concat output
_SDXL_POOLED_SIZE = 1280  # SDXL pooled text embed size
_SDXL_TIME_IDS = 6  # SDXL added_cond_kwargs time_ids length

# Compile-only mode uses a minimal cross_attention_dim to keep TT-MLIR
# compilation tractable (full 2048 produces 2048x160 K/V projection matrices
# that take 1.5+ hours to compile regardless of parameter count).
_COMPILE_ONLY_CROSS_ATTN_DIM = 64


class ModelVariant(StrEnum):
    """Available Amanatsu Illustrious model variants."""

    AMANATSU_ILLUSTRIOUS_V11 = "amanatsu-illustrious-v11"


class ModelLoader(ForgeModel):
    """Amanatsu Illustrious SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.AMANATSU_ILLUSTRIOUS_V11: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AMANATSU_ILLUSTRIOUS_V11

    prompt = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Amanatsu Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Amanatsu Illustrious SDXL pipeline.

        When TT_RANDOM_WEIGHTS=1, loads only the UNet config and uses random
        weights to avoid downloading the full 6.5GB model on compile-only systems.
        Uses float32 in that mode because AMD CPUs without AVX512BF16 fall back
        to a very slow bfloat16 conv2d path.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        from diffusers import UNet2DConditionModel

        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            # float32 is ~10x faster than bfloat16 on AMD EPYC (no AVX512BF16).
            # Scale down the architecture (block channels, transformer layers,
            # cross_attention_dim, addition_embed_type) to reduce TT-MLIR
            # compilation time from hours to seconds.  load_inputs() uses
            # matching shapes for the reduced config.
            unet_config = UNet2DConditionModel.load_config(
                self._variant_config.pretrained_model_name, subfolder="unet"
            )
            unet_config["block_out_channels"] = [160, 160, 160]
            unet_config["transformer_layers_per_block"] = [1, 1, 1]
            unet_config["layers_per_block"] = 1
            # Reduce cross_attention_dim from 2048 to avoid 2048×160 K/V
            # projection matrices that dominate TT-MLIR compile time (1.5h+).
            unet_config["cross_attention_dim"] = _COMPILE_ONLY_CROSS_ATTN_DIM
            # Remove text_time conditioning (64-head attention over 2816-dim
            # inputs) — also a major compile-time contributor.
            unet_config["addition_embed_type"] = None
            unet_config["projection_class_embeddings_input_dim"] = None
            unet = UNet2DConditionModel.from_config(unet_config)
            unet = unet.to(torch.float32).eval()
            return unet

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name, dtype=dtype
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Amanatsu Illustrious UNet model.

        Uses 512x512 resolution to keep the latent size (64x64) tractable for
        CPU reference runs. When TT_RANDOM_WEIGHTS=1, uses random float32 tensors
        with the correct SDXL input shapes (float32 avoids the slow bfloat16
        conv2d path on AMD CPUs without AVX512BF16).

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            # Use float32 — AMD EPYC 7352 has no AVX512BF16, making bfloat16
            # conv2d use a very slow fallback path.
            # CFG doubles the batch dimension.
            cfg_batch = batch_size * 2
            torch.manual_seed(42)
            # encoder_hidden_states uses _COMPILE_ONLY_CROSS_ATTN_DIM (64) to
            # match the reduced cross_attention_dim in the compile-only UNet.
            # added_cond_kwargs is omitted because addition_embed_type=None in
            # the compile-only config.
            return {
                "sample": torch.randn(
                    cfg_batch,
                    _SDXL_LATENT_CHANNELS,
                    _SDXL_LATENT_HEIGHT,
                    _SDXL_LATENT_WIDTH,
                    dtype=torch.float32,
                ),
                "timestep": torch.tensor(999, dtype=torch.float32),
                "encoder_hidden_states": torch.randn(
                    cfg_batch,
                    _SDXL_SEQ_LEN,
                    _COMPILE_ONLY_CROSS_ATTN_DIM,
                    dtype=torch.float32,
                ),
            }

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(
            self.pipeline, self.prompt, height=512, width=512
        )

        timestep = timesteps[0]

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": timestep.to(dtype),
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "added_cond_kwargs": added_cond_kwargs,
        }
