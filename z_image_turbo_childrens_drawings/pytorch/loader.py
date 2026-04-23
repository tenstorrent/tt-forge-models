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
from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image

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
        """Load and return the ZImageTransformer2DModel (a torch.nn.Module).

        Returns:
            ZImageTransformer2DModel: The transformer module.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        if os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC") or os.environ.get(
            "TT_RANDOM_WEIGHTS"
        ):
            self.transformer = self._load_transformer_with_random_weights(dtype)
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

    def _load_transformer_with_random_weights(self, dtype):
        """Build only the transformer from config with random weights for compile-only testing."""
        from diffusers import ZImageTransformer2DModel

        repo_id = self._variant_config.pretrained_model_name
        transformer_config = ZImageTransformer2DModel.load_config(
            repo_id, subfolder="transformer"
        )
        transformer = ZImageTransformer2DModel.from_config(transformer_config)
        if dtype != torch.float32:
            transformer = transformer.to(dtype)
        return transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the ZImageTransformer2DModel forward pass.

        Returns:
            list: [x, t, cap_feats] positional args for the transformer.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        in_channels = self.transformer.config.in_channels  # 16
        cap_feat_dim = self.transformer.config.cap_feat_dim  # 2560

        # One latent image per batch item: (C, F, H, W) with F=1 for image-mode
        latent = torch.randn(in_channels, 1, 64, 64, dtype=dtype)
        x = [latent] * batch_size

        # Batch of timesteps in [0, 1]
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # One text feature tensor per batch item: (seq_len, cap_feat_dim)
        cap = torch.randn(32, cap_feat_dim, dtype=dtype)
        cap_feats = [cap] * batch_size

        return [x, t, cap_feats]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack ZImageTransformer2DModel output to a single tensor."""
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output
        if isinstance(fwd_output, (tuple, list)):
            inner = fwd_output[0]
            if isinstance(inner, (list, tuple)):
                return inner[0]
            return inner
        if hasattr(fwd_output, "sample"):
            sample = fwd_output.sample
            if isinstance(sample, (list, tuple)):
                return sample[0]
            return sample
        return fwd_output
