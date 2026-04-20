# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViCLIP model loader implementation for video-text retrieval.

ViCLIP is a video foundation model based on CLIP ViT-L/14 that aligns video
clips and text in a shared embedding space. This loader exposes the vision
encoder path so the model can be driven with synthetic video tensors without
requiring the BPE tokenizer asset at runtime.
"""
from typing import Optional

import torch
import torch.nn as nn
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


class ViCLIPVisionWrapper(nn.Module):
    """Wraps ViCLIP to expose its vision encoder path as the forward pass.

    ViCLIP's native forward signature targets contrastive training and depends
    on loss modules that are not instantiated at inference time. This wrapper
    runs `encode_vision` in test mode, returning the pooled video embedding.
    """

    def __init__(self, viclip):
        super().__init__()
        self.viclip = viclip

    def forward(self, pixel_values):
        return self.viclip.encode_vision(pixel_values, test=True)


class ModelVariant(StrEnum):
    """Available ViCLIP model variants."""

    LARGE_PATCH14 = "Large_Patch14"


class ModelLoader(ForgeModel):
    """ViCLIP model loader implementation for PyTorch."""

    _VARIANTS = {
        ModelVariant.LARGE_PATCH14: ModelConfig(
            pretrained_model_name="OpenGVLab/ViCLIP-L-14-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PATCH14

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ViCLIP",
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

        viclip = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model = ViCLIPVisionWrapper(viclip)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # ViCLIP-L/14 expects 8 frames at 224x224.
        num_frames = 8
        height = 224
        width = 224
        channels = 3

        pixel_values = torch.randn(batch_size, num_frames, channels, height, width)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"pixel_values": pixel_values}
