# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro multimodal-understanding (image+text -> text) component loader.

Components match the deepseek-ai/Janus ``inference.py`` understanding loop:
  - Understand_1B / Understand_7B: language_model.model + lm_head over the
    combined image+text inputs_embeds (the compute-dominant LLaMA forward).
  - VisionEmbed_1B / VisionEmbed_7B: SigLIP vision tower + understanding aligner
    (pixel_values -> image embeddings in language space).
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


class ModelVariant(StrEnum):
    """Loadable understanding components for Janus-Pro 1B / 7B."""

    UNDERSTAND_1B = "Understand_1B"
    UNDERSTAND_7B = "Understand_7B"
    VISION_EMBED_1B = "VisionEmbed_1B"
    VISION_EMBED_7B = "VisionEmbed_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro multimodal-understanding components (deepseek-ai Janus runtime)."""

    _VARIANTS = {
        ModelVariant.UNDERSTAND_1B: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.UNDERSTAND_7B: ModelConfig(pretrained_model_name=REPO_ID_PRO_7B),
        ModelVariant.VISION_EMBED_1B: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.VISION_EMBED_7B: ModelConfig(pretrained_model_name=REPO_ID_PRO_7B),
    }
    DEFAULT_VARIANT = ModelVariant.UNDERSTAND_1B

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

    def _is_vision_embed(self) -> bool:
        return self._variant in (
            ModelVariant.VISION_EMBED_1B,
            ModelVariant.VISION_EMBED_7B,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        from .src.model import JanusUnderstandPrefill, JanusVisionEmbed
        from .src.model_utils import DTYPE, load_mmgpt

        dtype = dtype_override if dtype_override is not None else DTYPE
        mmgpt = load_mmgpt(self._repo_id(), dtype, **kwargs)

        if self._is_vision_embed():
            return JanusVisionEmbed(mmgpt.vision_model, mmgpt.aligner).eval()
        return JanusUnderstandPrefill(mmgpt).eval()

    def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        from .src.model_utils import (
            DTYPE,
            make_understanding_inputs_embeds,
            make_vision_embed_inputs,
        )

        dtype = dtype_override if dtype_override is not None else DTYPE
        repo_id = self._repo_id()

        if self._is_vision_embed():
            return make_vision_embed_inputs(repo_id, dtype)
        return make_understanding_inputs_embeds(repo_id, dtype)
