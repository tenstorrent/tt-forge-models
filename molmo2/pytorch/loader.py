# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2-8B component loader.

Molmo2 is a vision-language model (``Molmo2ForConditionalGeneration``) built from
a SigLIP-style vision tower + adapter and a Qwen3-8B-style text decoder. Each
variant brings up one compute component independently:

  - VISION_TOWER  -> Molmo2VisionTransformer (image ViT), single forward pass
  - TEXT_DECODER  -> Molmo2TextModel + lm_head (causal LM), single forward pass
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model_utils import (
    DTYPE,
    REPO_ID,
    load_text_decoder,
    load_text_decoder_inputs,
    load_vision_tower,
    load_vision_tower_inputs,
)


class ModelVariant(StrEnum):
    """Independently loadable Molmo2 compute components."""

    VISION_TOWER = "vision_tower"
    TEXT_DECODER = "text_decoder"


class ModelLoader(ForgeModel):
    """Load individual Molmo2 components without driving the full VLM pipeline."""

    _VARIANTS = {
        ModelVariant.VISION_TOWER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TEXT_DECODER: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TEXT_DECODER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if variant == ModelVariant.VISION_TOWER:
            task = ModelTask.CV_IMAGE_FE
        else:
            task = ModelTask.NLP_CAUSAL_LM
        return ModelInfo(
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            VISION_TOWER -> VisionTowerWrapper (image ViT)
            TEXT_DECODER -> TextDecoderWrapper (text decoder + lm_head)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE
        if self._variant == ModelVariant.VISION_TOWER:
            return load_vision_tower(dtype)
        if self._variant == ModelVariant.TEXT_DECODER:
            return load_text_decoder(dtype)
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a dict of sample inputs for the active component.

        VISION_TOWER -> {pixel_values [1, 729, 588]}
        TEXT_DECODER -> {input_ids [1, 32], attention_mask [1, 32]}
        """
        dtype = dtype_override if dtype_override is not None else DTYPE
        if self._variant == ModelVariant.VISION_TOWER:
            return load_vision_tower_inputs(dtype)
        if self._variant == ModelVariant.TEXT_DECODER:
            return load_text_decoder_inputs(dtype)
        raise ValueError(f"Unknown variant: {self._variant}")
