# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 vision-tower (image_vit) loader.

Molmo2-8B is a custom ``trust_remote_code`` VLM (Qwen3-8B text decoder +
SigLIP-style ViT). The full ``Molmo2ForConditionalGeneration`` forward contains
``.item()`` graph breaks and data-dependent adapter gathers, so — like other VLMs
brought up here — it is split into its device-compilable components. This loader
brings up the vision tower (``model.vision_backbone.image_vit``,
``Molmo2VisionTransformer``): a SigLIP-style ViT that maps patchified image
tokens ``[B, 729, 588]`` to per-patch features ``[B, 729, 1152]``.

See the sibling ``causal_lm`` loader for the text decoder, and ``.._compat`` for
the three transformers>=5.5 fixes both components share.
"""

import math
from typing import Optional

import torch

from ..._compat import apply_molmo2_compat, fix_rotary_inv_freq
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


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class _Molmo2VisionWrapper(torch.nn.Module):
    """Wraps ``image_vit`` so only the vision tower moves to device.

    Returns the final per-patch feature map (the last hidden state) as a single
    tensor for a clean component-level PCC comparison.
    """

    def __init__(self, image_vit):
        super().__init__()
        self.image_vit = image_vit

    def forward(self, pixel_values):
        return self.image_vit(pixel_values)[-1]


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B vision tower (SigLIP-style ViT)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # 27x27 patches of 14x14x3 = 729 tokens of dim 588 (SigLIP patchified input).
    _NUM_PATCHES = 729
    _PATCH_DIM = 588

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the Molmo2 vision tower wrapped so only it moves to device.

        Args:
            dtype_override: Optional torch.dtype (the runner passes bfloat16).

        Returns:
            torch.nn.Module: vision-tower wrapper returning ``[B, 729, 1152]``.
        """
        from transformers import AutoModelForImageTextToText

        # transformers>=5.5 fixes: re-register 'default' RoPE + neutralize the
        # int64-cumsum packed-sequence helper (must precede construction).
        apply_molmo2_compat()

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        full = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        # Recompute the corrupted rotary inv_freq buffers post-load (harmless for
        # the vision tower, which has none, but keeps the shared path uniform).
        fix_rotary_inv_freq(full)

        model = _Molmo2VisionWrapper(full.model.vision_backbone.image_vit)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build patchified SigLIP-normalized image tokens ``[B, 729, 588]``.

        A smooth, structured synthetic image (sin/cos gradients) is well
        conditioned and avoids the white-noise PCC hit that random input causes
        under bf16 — no network / real-image dependency.
        """
        grid = torch.linspace(0, 2 * math.pi, self._NUM_PATCHES).unsqueeze(1)
        feat = torch.linspace(0, 4 * math.pi, self._PATCH_DIM).unsqueeze(0)
        # SigLIP normalization uses mean=std=0.5, so well-formed input lies in
        # roughly [-1, 1]; keep the synthetic image comfortably inside that.
        image = 0.5 * torch.sin(grid) * torch.cos(feat)
        pixel_values = image.unsqueeze(0).repeat(batch_size, 1, 1)

        dtype = dtype_override if dtype_override is not None else torch.float32
        pixel_values = pixel_values.to(dtype)

        return {"pixel_values": pixel_values}
