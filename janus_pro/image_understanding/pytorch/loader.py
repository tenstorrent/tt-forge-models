# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro image-understanding component loader (deepseek-ai/Janus reference).

Complements the ``janus_pro/text_to_image`` generation loader with the
*understanding* path of the unified model:

  - vision_model: SigLIP CLIPVisionTower (image -> patch features)
  - aligner:      MlpProjector (features -> language-model embedding space)

The two together form the image encoder used by ``prepare_inputs_embeds`` to splice
image tokens into the text prompt. Weight loading, the transformers-5.x compat
patches, and the processor all come from the shared ``text_to_image`` runtime helpers
so the two components stay in lockstep.
"""

from __future__ import annotations

from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID_PRO_1B = "deepseek-ai/Janus-Pro-1B"
REPO_ID_PRO_7B = "deepseek-ai/Janus-Pro-7B"

# Bundled teaser image shipped with the checkpoint repo; used as a deterministic,
# always-available understanding input (no external dataset dependency).
_DEFAULT_IMAGE_FILE = "janus_pro_teaser1.png"
IMG_SIZE = 384


class ModelVariant(StrEnum):
    """Understanding image encoder, per checkpoint size."""

    UNDERSTAND_1B = "Understand_1B"
    UNDERSTAND_7B = "Understand_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro understanding image encoder (SigLIP vision_model + aligner)."""

    _VARIANTS = {
        ModelVariant.UNDERSTAND_1B: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.UNDERSTAND_7B: ModelConfig(pretrained_model_name=REPO_ID_PRO_7B),
    }
    DEFAULT_VARIANT = ModelVariant.UNDERSTAND_7B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus-Pro",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _repo_id(self) -> str:
        return self._variant_config.pretrained_model_name

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        # Reuse the shared text_to_image runtime: same weights, same compat patches.
        from ...text_to_image.pytorch.src.model_utils import DTYPE, load_mmgpt
        from .src.model import JanusUnderstandImageEncoder

        dtype = dtype_override if dtype_override is not None else DTYPE
        mmgpt = load_mmgpt(self._repo_id(), dtype, **kwargs)
        return JanusUnderstandImageEncoder(mmgpt.vision_model, mmgpt.aligner).eval()

    def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        from ...text_to_image.pytorch.src.model_utils import DTYPE

        dtype = dtype_override if dtype_override is not None else DTYPE
        pixel_values = self._preprocess_default_image(dtype)
        return {"pixel_values": pixel_values}

    def _preprocess_default_image(self, dtype: torch.dtype) -> torch.Tensor:
        """Preprocess the bundled teaser image to [1, 3, 384, 384] via the VLM processor."""
        from ...text_to_image.pytorch.src.model_utils import load_processor

        repo_id = self._repo_id()
        processor = load_processor(repo_id)
        try:
            from huggingface_hub import hf_hub_download
            from PIL import Image

            path = hf_hub_download(repo_id, _DEFAULT_IMAGE_FILE)
            image = Image.open(path).convert("RGB")
            pixel_values = processor.image_processor([image], return_tensors="pt")[
                "pixel_values"
            ]
        except Exception:
            # Deterministic fallback if the bundled image is unavailable.
            generator = torch.Generator().manual_seed(0)
            pixel_values = torch.rand(
                1, 3, IMG_SIZE, IMG_SIZE, generator=generator
            ) * 2 - 1
        return pixel_values.to(dtype=dtype)
