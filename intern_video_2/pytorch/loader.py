# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVideo2 Stage2 model loader implementation for video foundation tasks.

InternVideo2 is a 6B-parameter video foundation model that learns joint
video-text representations. The Stage2 checkpoint produces vision features
and CLIP-aligned embeddings that can be used for zero-shot video-text
retrieval and video classification.
"""

from typing import Optional

import torch
from transformers import AutoModel

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available InternVideo2 model variants."""

    STAGE2_6B = "Stage2_6B"


class ModelLoader(ForgeModel):
    """InternVideo2 Stage2 model loader for video foundation tasks."""

    _VARIANTS = {
        ModelVariant.STAGE2_6B: ModelConfig(
            pretrained_model_name="OpenGVLab/InternVideo2-Stage2_6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STAGE2_6B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternVideo2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate a synthetic video tensor matching the expected input format.

        The model's vision encoder expects input shape
        ``(batch, channels, num_frames, height, width)`` with 8 frames at
        224x224 resolution.
        """
        num_frames = 8
        image_size = 224
        channels = 3

        pixel_values = torch.randn(
            batch_size, channels, num_frames, image_size, image_size
        )

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"x": pixel_values}
