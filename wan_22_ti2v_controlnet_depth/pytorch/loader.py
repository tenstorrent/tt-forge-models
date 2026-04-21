# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 TI2V 5B depth ControlNet model loader.

TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1 is a depth-conditioned ControlNet
for the Wan-AI/Wan2.2-TI2V-5B-Diffusers text+image-to-video pipeline. Inputs
are a latent video tensor (48 VAE channels) together with a pixel-space RGB
depth guidance video that is spatially/temporally downscaled by the control
encoder before being fused with the latent stream.

Repository: https://huggingface.co/TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1
"""

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
from .src.model_utils import load_controlnet_inputs, load_wan_controlnet


class ModelVariant(StrEnum):
    """Available Wan 2.2 TI2V 5B depth ControlNet variants."""

    WAN22_TI2V_5B_CONTROLNET_DEPTH_V1 = "2.2_TI2V_5B_Controlnet_Depth_v1"


class ModelLoader(ForgeModel):
    """Loader for TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B_CONTROLNET_DEPTH_V1: ModelConfig(
            pretrained_model_name="TheDenk/wan2.2-ti2v-5b-controlnet-depth-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B_CONTROLNET_DEPTH_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_22_TI2V_CONTROLNET_DEPTH",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self._controlnet = load_wan_controlnet(
            self._variant_config.pretrained_model_name, dtype
        )
        return self._controlnet

    def load_inputs(
        self, *, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._controlnet is None:
            self.load_model(dtype_override=dtype)
        return load_controlnet_inputs(self._controlnet, dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if hasattr(output, "sample"):
            output = output.sample
        if isinstance(output, tuple):
            return torch.stack(list(output), dim=0)
        return output
