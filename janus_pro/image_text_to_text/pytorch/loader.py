# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro multimodal-understanding component loader (deepseek-ai/Janus reference).

Janus-Pro is a unified understanding + generation model. The generation
(text-to-image) pathway is covered by the sibling ``janus_pro/text_to_image``
loader. This loader brings up the *understanding* (image -> text) pathway.

Components:
  - UndVision_7B: SigLIP vision tower + understanding aligner
    (``aligner(vision_model(pixel_values))``), i.e. the image-encoding half of
    ``MultiModalityCausalLM.prepare_inputs_embeds``. Produces the per-image
    language-model embeddings that are spliced into the text stream before the
    LLM prefill (the LLM backbone itself is the same one brought up by the
    text_to_image ``ImageTokenStep`` component).

Weight loading and the transformers-5.x compatibility patches are reused from
the text_to_image package so there is a single source of truth for them.
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

REPO_ID_PRO_7B = "deepseek-ai/Janus-Pro-7B"

# SigLIP understanding tower: 384x384 input, 16x16 patches -> 576 image tokens.
IMG_SIZE = 384


class ModelVariant(StrEnum):
    """Loadable understanding components."""

    UND_VISION_7B = "UndVision_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro understanding components (deepseek-ai Janus runtime package)."""

    _VARIANTS = {
        ModelVariant.UND_VISION_7B: ModelConfig(
            pretrained_model_name=REPO_ID_PRO_7B
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UND_VISION_7B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus-Pro",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _repo_id(self) -> str:
        return self._variant_config.pretrained_model_name

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        from ...text_to_image.pytorch.src.model_utils import DTYPE, load_mmgpt
        from .src.model import JanusUndVision

        dtype = dtype_override if dtype_override is not None else DTYPE
        mmgpt = load_mmgpt(self._repo_id(), dtype, **kwargs)
        return JanusUndVision(mmgpt.vision_model, mmgpt.aligner).eval()

    def load_inputs(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        from ...text_to_image.pytorch.src.model_utils import DTYPE

        dtype = dtype_override if dtype_override is not None else DTYPE

        # Deterministic, normalized-range pixel values: one 384x384 RGB image per
        # batch element. The vision encoder is deterministic, so a fixed-seed
        # tensor is a valid CPU-vs-device PCC input; normalized Janus pixels sit
        # in roughly [-1.8, 2.0], which randn(0, 1) covers.
        generator = torch.Generator().manual_seed(0)
        pixel_values = torch.randn(
            batch_size, 3, IMG_SIZE, IMG_SIZE, generator=generator
        ).to(dtype=dtype)
        return {"pixel_values": pixel_values}
