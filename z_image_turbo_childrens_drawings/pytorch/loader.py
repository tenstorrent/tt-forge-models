# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo Children's Drawings LoRA (ostris/z_image_turbo_childrens_drawings)
model loader implementation.

Loads the Tongyi-MAI/Z-Image-Turbo base pipeline and applies the
ostris/z_image_turbo_childrens_drawings LoRA adapter to stylize text-to-image
generations in a children's drawing aesthetic.

Available variants:
- Z_IMAGE_TURBO_CHILDRENS_DRAWINGS: Z-Image-Turbo with Children's Drawings LoRA weights applied
"""

import os
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, ZImageTransformer2DModel

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


BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
ADAPTER_REPO_ID = "ostris/z_image_turbo_childrens_drawings"

# Latent space dimensions for a 512x512 image (VAE 8x compression)
_LATENT_CHANNELS = 16
_LATENT_H = 64
_LATENT_W = 64
_CAP_FEAT_DIM = 2560  # Qwen3 hidden size used as cap_feat_dim in config
_CAP_SEQ_LEN = 32  # must be a multiple of 32 (SEQ_MULTI_OF)


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo Children's Drawings LoRA model variants."""

    Z_IMAGE_TURBO_CHILDRENS_DRAWINGS = "Z_Image_Turbo_Childrens_Drawings"


class ModelLoader(ForgeModel):
    """Z-Image-Turbo Children's Drawings LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_TURBO_CHILDRENS_DRAWINGS: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_TURBO_CHILDRENS_DRAWINGS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_Image_Turbo_Childrens_Drawings",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo transformer.

        When TT_RANDOM_WEIGHTS is set, loads only the transformer config and
        initializes with random weights to avoid downloading large model files.

        Returns:
            ZImageTransformer2DModel: The denoising transformer module.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = ZImageTransformer2DModel.load_config(
                self._variant_config.pretrained_model_name,
                subfolder="transformer",
            )
            self.transformer = ZImageTransformer2DModel.from_config(config)
            if dtype_override is not None:
                self.transformer = self.transformer.to(dtype_override)
        else:
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                **kwargs,
            )
            self.pipeline.load_lora_weights(
                ADAPTER_REPO_ID,
                weight_name="z_image_turbo_childrens_drawings.safetensors",
            )
            self.transformer = self.pipeline.transformer

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the ZImageTransformer2DModel forward method.

        Returns a dict matching the transformer's forward signature:
            x: list of image latent tensors, each (C, F, H, W)
            t: timestep tensor of shape (batch_size,)
            cap_feats: list of caption feature tensors, each (seq_len, cap_feat_dim)
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        x = [
            torch.randn(_LATENT_CHANNELS, 1, _LATENT_H, _LATENT_W, dtype=dtype)
            for _ in range(batch_size)
        ]
        t = torch.full((batch_size,), 0.5, dtype=dtype)
        cap_feats = [
            torch.randn(_CAP_SEQ_LEN, _CAP_FEAT_DIM, dtype=dtype)
            for _ in range(batch_size)
        ]

        return {"x": x, "t": t, "cap_feats": cap_feats}
