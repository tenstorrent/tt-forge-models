# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro text-to-image loader for the test_models single-device runner.

Compiles one T2I subgraph: language_model forward on CFG prompt embeds, then
generation_head (first image-token step, matching transformers image generate i=0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import JanusForConditionalGeneration, JanusProcessor

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
from .src.model import JanusImageTokenLogitsStep
from .src.model_utils import (
    STANDARD_PROMPT,
    load_cfg_prompt_inputs,
    load_janus_model,
    load_processor,
)


class ModelVariant(StrEnum):
    """Janus-Pro variants (transformers deepseek-community/*)."""

    PRO_1B = "Pro_1B"
    PRO_7B = "Pro_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro T2I bring-up loader for test_models.py (single_device inference)."""

    _VARIANTS = {
        ModelVariant.PRO_1B: ModelConfig(
            pretrained_model_name="deepseek-community/Janus-Pro-1B",
        ),
        ModelVariant.PRO_7B: ModelConfig(
            pretrained_model_name="deepseek-community/Janus-Pro-7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRO_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor: JanusProcessor | None = None
        self._model: JanusForConditionalGeneration | None = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Janus-Pro",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_processor(self) -> JanusProcessor:
        if self._processor is None:
            self._processor = load_processor(self._variant_config.pretrained_model_name)
        return self._processor

    def _get_backbone(
        self, dtype_override: torch.dtype | None, **kwargs
    ) -> JanusForConditionalGeneration:
        if self._model is None:
            self._model = load_janus_model(
                self._variant_config.pretrained_model_name,
                dtype_override=dtype_override,
                **kwargs,
            )
        return self._model

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the first image-token logits subgraph (not the full HF model)."""
        backbone = self._get_backbone(dtype_override, **kwargs)
        step = JanusImageTokenLogitsStep(backbone)
        if dtype_override is not None:
            step = step.to(dtype=dtype_override)
        return step

    def load_inputs(self, dtype_override=None):
        """CFG-doubled prompt embeds for the compiled subgraph (fixed batch dim 2)."""
        backbone = self._get_backbone(dtype_override)
        processor = self._get_processor()
        return load_cfg_prompt_inputs(
            backbone,
            processor,
            dtype_override,
            prompt=STANDARD_PROMPT,
        )
