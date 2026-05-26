# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro text-to-image component loader (deepseek-ai/Janus reference loop).

Components match generation_inference.py:
  - ImageTokenStep / Pro_1B / Pro_7B: language_model.model + gen_head (KV optional)
  - GenImgEmbed: gen_embed + gen_aligner
  - GenVisionDecode: gen_vision_model.decode_code
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
    """Loadable components; Pro_1B/Pro_7B alias ImageTokenStep weights."""

    IMAGE_TOKEN_STEP = "ImageTokenStep"
    GEN_IMG_EMBED = "GenImgEmbed"
    GEN_IMG_EMBED_7B = "GenImgEmbed_7B"
    GEN_VISION_DECODE = "GenVisionDecode"
    GEN_VISION_DECODE_7B = "GenVisionDecode_7B"
    PRO_1B = "Pro_1B"
    PRO_7B = "Pro_7B"


class ModelLoader(ForgeModel):
    """Janus-Pro T2I components via the janus package (no HF generate())."""

    _VARIANTS = {
        ModelVariant.PRO_1B: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.PRO_7B: ModelConfig(pretrained_model_name=REPO_ID_PRO_7B),
        ModelVariant.IMAGE_TOKEN_STEP: ModelConfig(
            pretrained_model_name=REPO_ID_PRO_1B
        ),
        ModelVariant.GEN_IMG_EMBED: ModelConfig(pretrained_model_name=REPO_ID_PRO_1B),
        ModelVariant.GEN_IMG_EMBED_7B: ModelConfig(
            pretrained_model_name=REPO_ID_PRO_7B
        ),
        ModelVariant.GEN_VISION_DECODE: ModelConfig(
            pretrained_model_name=REPO_ID_PRO_1B
        ),
        ModelVariant.GEN_VISION_DECODE_7B: ModelConfig(
            pretrained_model_name=REPO_ID_PRO_7B
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PRO_1B

    @classmethod
    def _resolve_component(cls, variant: ModelVariant) -> ModelVariant:
        if variant in (ModelVariant.PRO_1B, ModelVariant.PRO_7B):
            return ModelVariant.IMAGE_TOKEN_STEP
        if variant == ModelVariant.GEN_IMG_EMBED_7B:
            return ModelVariant.GEN_IMG_EMBED
        if variant == ModelVariant.GEN_VISION_DECODE_7B:
            return ModelVariant.GEN_VISION_DECODE
        return variant

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

    def _repo_id(self) -> str:
        return self._variant_config.pretrained_model_name

    def _component(self) -> ModelVariant:
        return self._resolve_component(self._variant)

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        from .src.model import (
            JanusGenImgEmbed,
            JanusGenVisionDecode,
            JanusGitImageTokenStep,
        )
        from .src.model_utils import DTYPE, decode_shape, load_mmgpt

        dtype = dtype_override if dtype_override is not None else DTYPE
        repo_id = self._repo_id()
        component = self._component()

        if component == ModelVariant.IMAGE_TOKEN_STEP:
            mmgpt = load_mmgpt(repo_id, dtype, **kwargs)
            return JanusGitImageTokenStep(mmgpt).eval()

        if component == ModelVariant.GEN_IMG_EMBED:
            mmgpt = load_mmgpt(repo_id, dtype, **kwargs)
            return JanusGenImgEmbed(mmgpt.gen_embed, mmgpt.gen_aligner).eval()

        if component == ModelVariant.GEN_VISION_DECODE:
            mmgpt = load_mmgpt(repo_id, dtype, **kwargs)
            return JanusGenVisionDecode(
                mmgpt.gen_vision_model, decode_shape()
            ).eval()

        raise ValueError(f"Unknown component: {component}")

    def load_inputs(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        prefill: bool = False,
        **kwargs,
    ):
        from .src.model_utils import (
            DTYPE,
            make_cfg_inputs_embeds,
            make_gen_img_embed_inputs,
            make_gen_vision_decode_inputs,
            make_image_token_decode_inputs,
        )

        dtype = dtype_override if dtype_override is not None else DTYPE
        repo_id = self._repo_id()
        component = self._component()

        if component == ModelVariant.IMAGE_TOKEN_STEP:
            if prefill:
                inputs_embeds = make_cfg_inputs_embeds(repo_id, dtype)
                return {"inputs_embeds": inputs_embeds}
            return make_image_token_decode_inputs(repo_id, dtype)

        if component == ModelVariant.GEN_IMG_EMBED:
            return make_gen_img_embed_inputs(repo_id, dtype)

        if component == ModelVariant.GEN_VISION_DECODE:
            return make_gen_vision_decode_inputs(repo_id, dtype)

        raise ValueError(f"Unknown component: {component}")
