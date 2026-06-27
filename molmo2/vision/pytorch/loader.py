# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Molmo2 vision-tower loader (SigLIP-style image encoder).

Brings up the image encoder of ``allenai/Molmo2-8B`` as a standalone forward
pass. The full ``Molmo2ForConditionalGeneration`` graph has ``.item()`` graph
breaks and data-dependent adapter gathers, so — like other VLMs here — we split
the model into its vision tower and text decoder and bring up each separately.
The vision tower is ``model.model.vision_backbone.image_vit``
(``Molmo2VisionTransformer``); it consumes patchified images of shape
``[B, 729, 588]`` (27x27 patches of 14x14x3) and returns per-patch features.
"""

import math
from typing import Optional

import torch

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
from ....tools.utils import cast_input_to_type
from ..._compat import register_default_rope, patch_packed_sequence_indices


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class _VisionTower(torch.nn.Module):
    """Wraps ``image_vit`` so the forward returns a single feature tensor.

    ``Molmo2VisionTransformer.forward`` returns the list of per-block hidden
    states; we return the final one (``[B, num_patches, hidden]``) for a clean
    single-tensor PCC comparison.
    """

    def __init__(self, image_vit: torch.nn.Module):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.image_vit(pixel_values)
        if isinstance(hidden_states, (list, tuple)):
            return hidden_states[-1]
        return hidden_states


class ModelLoader(ForgeModel):
    """Molmo2 vision-tower loader."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # 27x27 patches of 14x14x3 pixels -> [B, 729, 588]
    _NUM_PATCHES_SIDE = 27
    _PATCH_SIZE = 14
    _CHANNELS = 3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the Molmo2 vision tower for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to load weights in (e.g. bfloat16).

        Returns:
            torch.nn.Module: The wrapped vision tower (image_vit).
        """
        from transformers import AutoModelForImageTextToText

        pretrained_model_name = self._variant_config.pretrained_model_name

        # transformers >= 5.5 compatibility: the 'default' RoPE init was removed,
        # and constructing the full model builds the text decoder's rotary
        # embeddings, so register it before from_pretrained.
        register_default_rope()
        patch_packed_sequence_indices()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()
        self.config = full_model.config

        image_vit = full_model.model.vision_backbone.image_vit
        vision_tower = _VisionTower(image_vit)
        vision_tower.eval()
        if dtype_override is not None:
            vision_tower = vision_tower.to(dtype_override)
        return vision_tower

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build a well-conditioned synthetic patchified image.

        A smooth, structured image (sinusoidal gradients, SigLIP mean=std=0.5
        normalization) avoids the PCC hit that white noise causes under bf16,
        without needing any network / real-image dependency.

        Args:
            dtype_override: Optional torch.dtype for the pixel values.
            batch_size: Batch size for the inputs.

        Returns:
            dict: {"pixel_values": tensor of shape [B, 729, 588]}.
        """
        side = self._NUM_PATCHES_SIDE * self._PATCH_SIZE  # 378
        ys = torch.linspace(0.0, 1.0, side).view(side, 1)
        xs = torch.linspace(0.0, 1.0, side).view(1, side)
        # Three smooth channels at different spatial frequencies, in [0, 1].
        ch0 = 0.5 + 0.5 * torch.sin(2 * math.pi * (2.0 * xs + 1.0 * ys))
        ch1 = 0.5 + 0.5 * torch.cos(2 * math.pi * (1.0 * xs + 2.0 * ys))
        ch2 = 0.5 + 0.5 * torch.sin(2 * math.pi * (3.0 * (xs * ys)))
        img = torch.stack([ch0.expand(side, side), ch1.expand(side, side), ch2], dim=0)
        img = img.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 3, 378, 378]

        # SigLIP normalization: mean = std = 0.5 -> maps [0, 1] to [-1, 1].
        img = (img - 0.5) / 0.5

        # Patchify into [B, num_patches, patch_size*patch_size*channels].
        B = batch_size
        C, P, H = self._CHANNELS, self._PATCH_SIZE, self._NUM_PATCHES_SIDE
        x = img.reshape(B, C, H, P, H, P)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H, W, P, P, C]
        pixel_values = x.reshape(B, H * H, P * P * C)  # [B, 729, 588]

        pixel_values = cast_input_to_type(pixel_values, dtype_override)
        return {"pixel_values": pixel_values}
