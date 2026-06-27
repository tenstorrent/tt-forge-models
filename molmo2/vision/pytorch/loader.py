# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 vision-tower loader implementation.

Molmo2 (allenai/Molmo2-8B) is a multimodal image-text-to-text model. Its image
encoder is a SigLIP-style vision transformer (``Molmo2VisionTransformer``): a
linear patch embedding, learned positional embeddings, and a stack of pre-norm
transformer blocks. This loader brings up that vision tower (the part of a VLM
most likely to block on device) as a single forward pass returning the final
hidden state.

The downstream image-pooling / projector path (``Molmo2VisionBackbone.forward``)
uses boolean-mask gather producing a data-dependent output shape, which the
static-shape device path does not support, so it is intentionally excluded here.

The model ships as custom HuggingFace code (``trust_remote_code=True``); the
revision is pinned for reproducibility.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoConfig

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class _VisionTowerWrapper(nn.Module):
    """Wrap Molmo2VisionTransformer to return the final hidden state tensor."""

    def __init__(self, image_vit: nn.Module):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_patches: torch.Tensor) -> torch.Tensor:
        # Molmo2VisionTransformer returns a list of per-layer hidden states.
        hidden_states = self.image_vit(pixel_patches)
        return hidden_states[-1]


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 vision-tower (image encoder) loader."""

    # Pinned revision of the custom-code repo for reproducibility.
    _REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: ModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.config = None
        self._vit_config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="molmo2_vision",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Load the Molmo2 vision tower (image_vit) for this instance's variant.

        Loads the full Molmo2ForConditionalGeneration checkpoint and extracts the
        vision transformer submodule, wrapping it to return a single tensor.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.

        Returns:
            torch.nn.Module: The wrapped Molmo2 vision tower.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True, "revision": self._REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = full_model.config
        image_vit = full_model.model.vision_backbone.image_vit
        self._vit_config = image_vit.config

        model = _VisionTowerWrapper(image_vit)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs (patchified pixel values) for the vision tower.

        Args:
            dtype_override: Optional torch.dtype for the input tensor.
            batch_size: Optional batch size (number of image crops). Default 1.

        Returns:
            dict: {"pixel_patches": tensor of shape (batch, num_patch, n_pixels)}.
        """
        if self._vit_config is None:
            cfg = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                revision=self._REVISION,
            )
            vit_config = cfg.vit_config
        else:
            vit_config = self._vit_config

        num_patch = vit_config.image_num_pos  # 729 at native 27x27 grid
        n_pixels = vit_config.image_patch_size * vit_config.image_patch_size * 3

        # Deterministic synthetic patches sized to the native input grid.
        gen = torch.Generator().manual_seed(0)
        pixel_patches = torch.randn(
            batch_size, num_patch, n_pixels, generator=gen, dtype=torch.float32
        )

        if dtype_override is not None:
            pixel_patches = pixel_patches.to(dtype_override)

        return {"pixel_patches": pixel_patches}

    def load_config(self):
        """Load and return the configuration for the Molmo2 variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=self._REVISION,
        )
        return self.config
