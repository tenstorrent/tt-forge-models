# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoPrism LvT model loader implementation for JAX.

VideoPrism is a foundational video encoder that jointly embeds video and text.
This loader supports the LvT (Language-video Transformer) variant which combines
a ViViT video encoder with a CoCa-initialized text encoder.

Repository:
- https://huggingface.co/google/videoprism-lvt-base-f16r288
- https://huggingface.co/google/videoprism-lvt-large-f8r288
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

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


@dataclass
class VideoPrismConfig(ModelConfig):
    """Configuration specific to VideoPrism LvT models."""

    vp_model_name: str = ""
    num_frames: int = 16
    resolution: int = 288


class ModelVariant(StrEnum):
    """Available VideoPrism LvT model variants."""

    LVT_BASE_F16R288 = "LvT_Base_F16R288"
    LVT_LARGE_F8R288 = "LvT_Large_F8R288"


class ModelLoader(ForgeModel):
    """VideoPrism LvT model loader implementation for JAX."""

    _VARIANTS = {
        ModelVariant.LVT_BASE_F16R288: VideoPrismConfig(
            pretrained_model_name="google/videoprism-lvt-base-f16r288",
            vp_model_name="videoprism_lvt_public_v1_base",
            num_frames=16,
            resolution=288,
        ),
        ModelVariant.LVT_LARGE_F8R288: VideoPrismConfig(
            pretrained_model_name="google/videoprism-lvt-large-f8r288",
            vp_model_name="videoprism_lvt_public_v1_large",
            num_frames=8,
            resolution=288,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LVT_BASE_F16R288

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None
        self._state = None
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="VideoPrism",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the VideoPrism LvT model.

        Returns:
            The VideoPrism Flax model instance.
        """
        from videoprism import models as vp

        model_name = self._variant_config.vp_model_name
        self._model = vp.get_model(model_name)
        self._state = vp.load_pretrained_weights(model_name)
        self._tokenizer = vp.load_text_tokenizer("c4_en")

        return self._model

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load sample inputs for the VideoPrism LvT model.

        Returns:
            dict: Dictionary with video_inputs, text_token_ids, and text_token_paddings.
        """
        from videoprism import models as vp

        dtype = dtype_override if dtype_override is not None else jnp.float32

        num_frames = self._variant_config.num_frames
        resolution = self._variant_config.resolution

        # Video input: [batch_size, num_frames, height, width, channels]
        video_inputs = jnp.zeros(
            (1, num_frames, resolution, resolution, 3), dtype=dtype
        )

        # Tokenize a sample text query
        text_queries = ["a video of a cat"]
        text_token_ids, text_token_paddings = vp.tokenize_texts(
            self._tokenizer, text_queries
        )

        return {
            "inputs": video_inputs,
            "text_token_ids": text_token_ids,
            "text_token_paddings": text_token_paddings,
        }
