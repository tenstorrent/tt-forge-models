# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Virtual Camera model loader for tt_forge_models.

Stable Virtual Camera (SeVa) is a 1.3B parameter diffusion model for novel view
synthesis. It generates 3D-consistent novel views of a scene from input images
and specified camera trajectories at 576p resolution.

Repository: https://huggingface.co/stabilityai/stable-virtual-camera
"""

import os
from typing import Any, Optional

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


class ModelVariant(StrEnum):
    """Available Stable Virtual Camera variants."""

    V1_0 = "v1.0"
    V1_1 = "v1.1"


class ModelLoader(ForgeModel):
    """
    Loader for Stable Virtual Camera (SeVa) novel view synthesis model.

    The model is a U-Net-style diffusion model with multi-view transformer
    attention that generates novel camera viewpoints from input images.
    """

    _VARIANTS = {
        ModelVariant.V1_0: ModelConfig(
            pretrained_model_name="stabilityai/stable-virtual-camera",
        ),
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name="stabilityai/stable-virtual-camera",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1

    _WEIGHT_NAMES = {
        ModelVariant.V1_0: "model.safetensors",
        ModelVariant.V1_1: "modelv1.1.safetensors",
    }

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="StableVirtualCamera",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from seva.model import Seva, SevaParams

        model = Seva(SevaParams())

        if not os.environ.get("TT_RANDOM_WEIGHTS"):
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            weight_name = self._WEIGHT_NAMES[self._variant]
            weight_path = hf_hub_download(
                self._variant_config.pretrained_model_name,
                filename=weight_name,
            )
            state_dict = load_file(weight_path)
            model.load_state_dict(state_dict)

        return model.eval()

    def load_inputs(self, *, dtype_override=None, **kwargs) -> dict:
        from seva.model import SevaParams

        params = SevaParams()

        batch_frames = params.num_frames
        height = 72
        width = 72

        x = torch.randn(batch_frames, params.in_channels, height, width)
        t = torch.ones(batch_frames)
        y = torch.randn(batch_frames, 1, params.context_dim)
        dense_y = torch.randn(batch_frames, params.dense_in_channels, height, width)

        return {
            "x": x,
            "t": t,
            "y": y,
            "dense_y": dense_y,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
