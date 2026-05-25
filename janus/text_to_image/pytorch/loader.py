# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro text-to-image loader (Path A — deepseek-ai/Janus git pipeline).

Compiles step-0 subgraph: CFG prompt embeds -> JanusGitImageTokenStep0 -> pre-CFG logits.
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
from .src.model import JanusGitImageTokenStep0
from .src.model_utils import (
    DTYPE,
    REPO_ID_PRO_1B,
    REPO_ID_PRO_7B,
    load_mmgpt,
    make_cfg_inputs_embeds,
)


class ModelVariant(StrEnum):
    PRO_1B = "Pro_1B"
    PRO_7B = "Pro_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro T2I bring-up: generation_inference.py step-0 via janus package."""

    _VARIANTS = {
        ModelVariant.PRO_1B: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.PRO_7B: ModelConfig(pretrained_model_name=REPO_ID_PRO_7B),
    }
    DEFAULT_VARIANT = ModelVariant.PRO_1B

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

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE
        repo_id = self._variant_config.pretrained_model_name
        mmgpt = load_mmgpt(repo_id, dtype, **kwargs)
        return JanusGitImageTokenStep0(mmgpt).eval()

    def load_inputs(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE
        repo_id = self._variant_config.pretrained_model_name
        inputs_embeds = make_cfg_inputs_embeds(repo_id, dtype)
        return {"inputs_embeds": inputs_embeds}
