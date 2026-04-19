# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD3.5 ControlNets ComfyUI Repackaged model loader implementation.

Loads single-file safetensors ControlNet variants from
Comfy-Org/stable-diffusion-3.5-controlnets_ComfyUI_repackaged.
Uses upstream stabilityai diffusers configs for model construction.

Available variants:
- SD35_CONTROLNET_BLUR: SD3.5 Large ControlNet Blur
- SD35_CONTROLNET_CANNY: SD3.5 Large ControlNet Canny
- SD35_CONTROLNET_DEPTH: SD3.5 Large ControlNet Depth
"""

from typing import Any, Optional

import torch
from diffusers import SD3ControlNetModel

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

_REPO_IDS = {
    "blur": "stabilityai/stable-diffusion-3.5-large-controlnet-blur",
    "canny": "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
    "depth": "stabilityai/stable-diffusion-3.5-large-controlnet-depth",
}


class ModelVariant(StrEnum):
    """Available SD3.5 ControlNet ComfyUI model variants."""

    SD35_CONTROLNET_BLUR = "ControlNet_Blur"
    SD35_CONTROLNET_CANNY = "ControlNet_Canny"
    SD35_CONTROLNET_DEPTH = "ControlNet_Depth"


class ModelLoader(ForgeModel):
    """SD3.5 ControlNets ComfyUI Repackaged model loader."""

    _VARIANTS = {
        ModelVariant.SD35_CONTROLNET_BLUR: ModelConfig(
            pretrained_model_name=_REPO_IDS["blur"],
        ),
        ModelVariant.SD35_CONTROLNET_CANNY: ModelConfig(
            pretrained_model_name=_REPO_IDS["canny"],
        ),
        ModelVariant.SD35_CONTROLNET_DEPTH: ModelConfig(
            pretrained_model_name=_REPO_IDS["depth"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SD35_CONTROLNET_CANNY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD3_5_CONTROLNETS_COMFYUI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        """Map variant to version key for file/config lookup."""
        return {
            ModelVariant.SD35_CONTROLNET_BLUR: "blur",
            ModelVariant.SD35_CONTROLNET_CANNY: "canny",
            ModelVariant.SD35_CONTROLNET_DEPTH: "depth",
        }[self._variant]

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> SD3ControlNetModel:
        """Load ControlNet from upstream stabilityai diffusers repo."""
        version = self._get_version_key()
        self._controlnet = SD3ControlNetModel.from_pretrained(
            _REPO_IDS[version],
            torch_dtype=dtype,
        )
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the SD3.5 ControlNet model.

        Returns:
            SD3ControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the SD3.5 ControlNet.

        Returns a dict matching SD3ControlNetModel.forward() signature.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        config = self._controlnet.config
        inner_dim = config.num_attention_heads * config.attention_head_dim
        seq_len = (config.sample_size // config.patch_size) ** 2

        hidden_states = torch.randn(batch_size, seq_len, inner_dim, dtype=dtype)
        pooled_projections = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        controlnet_cond = torch.randn(
            batch_size,
            config.in_channels,
            config.sample_size,
            config.sample_size,
            dtype=dtype,
        )

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": 1.0,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
