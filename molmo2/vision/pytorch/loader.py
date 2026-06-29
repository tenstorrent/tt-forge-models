# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 vision-tower loader implementation (image feature extraction).

This loader brings up the *vision tower* of Molmo2-8B as a single forward pass
for an op-support pre-check on Tenstorrent hardware. It extracts the SigLIP-style
vision transformer (``Molmo2VisionTransformer``, the ``image_vit`` submodule of
the model's vision backbone) from the full ``Molmo2ForConditionalGeneration``
checkpoint, so the weights are the real pretrained weights.

The full ``Molmo2VisionBackbone`` forward additionally performs data-dependent
pooling (boolean-mask token selection, clip+gather) that produces dynamic output
shapes and does not lower to the static-shape device path; those ops are why the
end-to-end multimodal forward is blocked on device. This loader isolates the pure
transformer (patch-embed Linear + positional embedding + attention blocks), which
is the device-compilable part of the vision path.

The model is custom-code on the Hub (``trust_remote_code=True``).

.. note::
    See ``molmo2/image_text_to_text/pytorch/loader.py``: the published remote code
    requires ``transformers==4.57.1`` and is incompatible with the transformers 5.x
    that the tt-forge device stack requires. Verified on CPU under 4.57.1; on-device
    bringup is blocked on this transformers-version conflict.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class _ViTLastHidden(nn.Module):
    """Wrap Molmo2VisionTransformer to return the last-layer hidden state.

    The underlying transformer returns a list of per-layer hidden states; for a
    single-tensor forward (and a clean PCC comparison) we return the final one.
    """

    def __init__(self, image_vit: nn.Module):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.image_vit(x)
        return hidden_states[-1]


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 vision-tower loader for image feature-extraction op pre-check."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.config = None
        self._num_patches = None
        self._patch_dim = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model info with validated variant."""
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Molmo2 vision transformer (image_vit) with real weights.

        Loads the full Molmo2ForConditionalGeneration checkpoint, extracts the
        vision backbone's image ViT, and discards the rest of the model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default.

        Returns:
            torch.nn.Module: The wrapped vision transformer.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()

        vision_backbone = full_model.model.vision_backbone
        image_vit = vision_backbone.image_vit
        vit_config = image_vit.config

        # Record input geometry: patches = image_num_pos, pixels-per-patch = patch^2 * 3.
        self.config = vit_config
        self._num_patches = vit_config.image_num_pos
        self._patch_dim = vit_config.image_patch_size * vit_config.image_patch_size * 3

        model = _ViTLastHidden(image_vit)
        model.eval()

        # Release the rest of the full model (decoder, lm_head, adapter, embeddings).
        del full_model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a deterministic image-patch tensor for the vision transformer.

        Shape: (batch, num_patches, patch_size^2 * 3) — i.e. flattened pixel
        values per patch, matching what the patch-embed Linear expects.

        Args:
            dtype_override: Optional torch.dtype for the pixel tensor.
            batch_size: Batch size for the inputs.

        Returns:
            dict: {"x": pixel-patch tensor}
        """
        if self._num_patches is None:
            # Geometry not yet resolved (load_model not called); use known config.
            self._num_patches = 729  # 378/14 = 27 -> 27*27
            self._patch_dim = 14 * 14 * 3  # 588

        gen = torch.Generator().manual_seed(0)
        x = torch.randn(
            batch_size,
            self._num_patches,
            self._patch_dim,
            generator=gen,
        )
        if dtype_override is not None:
            x = x.to(dtype_override)
        return {"x": x}
