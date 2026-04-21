# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V-Express V-KPS guider model loader implementation.

V-Express (Tencent AI Lab) is a portrait video generation pipeline that
produces talking-head videos conditioned on audio and facial keypoints.
This loader tests the VKpsGuider component, which encodes keypoint images
into a conditioning feature volume consumed by the denoising UNet.
Source: https://huggingface.co/tk93/V-Express
"""

from huggingface_hub import hf_hub_download
from typing import Optional

import torch

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
    """Available V-Express model variants."""

    V_KPS_GUIDER = "v_kps_guider"


class ModelLoader(ForgeModel):
    """V-Express V-KPS guider model loader implementation."""

    _VARIANTS = {
        ModelVariant.V_KPS_GUIDER: ModelConfig(
            pretrained_model_name="tk93/V-Express",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V_KPS_GUIDER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="V-Express",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.model_utils import load_v_kps_guider

        pretrained_model_name = self._variant_config.pretrained_model_name

        checkpoint_path = hf_hub_download(
            repo_id=pretrained_model_name,
            filename="v_kps_guider.bin",
        )

        model = load_v_kps_guider(checkpoint_path, device="cpu")

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # V-Express feeds a sequence of 512x512 RGB keypoint images through
        # the guider as a 5D video tensor (B, C, T, H, W).
        dtype = dtype_override or torch.float32
        inputs = torch.randn(batch_size, 3, 4, 512, 512, dtype=dtype)
        return inputs
