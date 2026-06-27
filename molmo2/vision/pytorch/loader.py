# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-8B vision-tower loader (SigLIP-style ViT image encoder).

The vision tower (``Molmo2VisionTransformer``, a.k.a. ``image_vit``) is the part
of allenai/Molmo2-8B most likely to block on device, so it is brought up as a
separate component from the text decoder (``molmo2/causal_lm/pytorch/loader.py``).

It consumes pre-patchified, SigLIP-normalized image patches of shape
``[B, num_patch=729, n_pixels=588]`` (27x27 patches of 14x14x3) directly — the
Molmo2 image *processor* rejects ``image_use_col_tokens`` under transformers>=5.5,
so the patch tensor is built in ``load_inputs`` rather than via the processor.

See ``molmo2/_compat.py``: the default-RoPE fix is needed to construct the full
model even though the ViT itself uses learned positional embeddings, not RoPE.
"""

import math
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ..._compat import register_default_rope

_MOLMO2_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

# SigLIP-style ViT geometry (from the Molmo2 vit_config).
_PATCH = 14
_GRID = 27  # 378 / 14 -> 27x27 patches
_NUM_PATCH = _GRID * _GRID  # 729
_N_PIXELS = _PATCH * _PATCH * 3  # 588


class ModelVariant(StrEnum):
    """Available Molmo2 vision-tower variants."""

    MOLMO2_8B = "8b"


class _Molmo2VisionTower(torch.nn.Module):
    """Runs the ViT and returns its final hidden state ([B, 729, 1152])."""

    def __init__(self, full_model):
        super().__init__()
        self.image_vit = full_model.model.vision_backbone.image_vit

    def forward(self, images):
        hidden_states = self.image_vit(images)  # list of per-block hidden states
        return hidden_states[-1]


class ModelLoader(ForgeModel):
    """Loader for the Molmo2-8B vision tower (image feature extraction)."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load the full Molmo2 model and return its vision tower."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Needed to construct the full model even though the ViT itself has no RoPE.
        register_default_rope()

        model_kwargs = {"trust_remote_code": True, "revision": _MOLMO2_REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        full_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()

        wrapper = _Molmo2VisionTower(full_model).eval()
        if dtype_override is not None:
            wrapper = wrapper.to(dtype_override)
        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Build SigLIP-normalized patchified image input [B, 729, 588].

        A smooth, structured (low-spatial-frequency) synthetic image is used so
        the input is well-conditioned for bf16 — white noise stresses the ViT's
        bf16 accumulation and deflates PCC, while smooth natural-image-like
        content keeps it high. CPU and TT see the identical tensor either way.
        """
        height = width = _GRID * _PATCH  # 378

        ys = torch.linspace(0, 1, height).view(height, 1)
        xs = torch.linspace(0, 1, width).view(1, width)
        red = 0.5 + 0.5 * torch.sin(6 * math.pi * xs) * torch.cos(4 * math.pi * ys)
        green = 0.5 + 0.5 * torch.sin(3 * math.pi * (xs + ys))
        blue = 0.5 + 0.5 * torch.cos(5 * math.pi * xs) * torch.sin(2 * math.pi * ys)
        image = torch.stack([red, green, blue], dim=0)  # [3, H, W] in [0, 1]

        # SigLIP normalization (mean=std=0.5) -> roughly [-1, 1].
        image = (image - 0.5) / 0.5

        # Patchify -> [num_patch, 14*14*3].
        image = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, 3, H, W]
        image = image.reshape(batch_size, 3, _GRID, _PATCH, _GRID, _PATCH)
        image = image.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, _NUM_PATCH, _N_PIXELS
        )

        if dtype_override is not None:
            image = image.to(dtype_override)

        return {"images": image}
