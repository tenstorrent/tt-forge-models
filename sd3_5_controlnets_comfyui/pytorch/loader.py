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

REPO_ID = "Comfy-Org/stable-diffusion-3.5-controlnets_ComfyUI_repackaged"

# ControlNet file paths within the ComfyUI repackaged repo
_CONTROLNET_FILES = {
    "blur": "split_files/controlnet/sd3.5_large_controlnet_blur.safetensors",
    "canny": "split_files/controlnet/sd3.5_large_controlnet_canny.safetensors",
    "depth": "split_files/controlnet/sd3.5_large_controlnet_depth.safetensors",
}

# Upstream diffusers config sources for each variant
_CONFIGS = {
    "blur": "stabilityai/stable-diffusion-3.5-large-controlnet-blur",
    "canny": "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
    "depth": "stabilityai/stable-diffusion-3.5-large-controlnet-depth",
}

# SD3.5 Large transformer config dimensions
SAMPLE_SIZE = 128
PATCH_SIZE = 2
IN_CHANNELS = 16
NUM_ATTENTION_HEADS = 38
ATTENTION_HEAD_DIM = 64
INNER_DIM = NUM_ATTENTION_HEADS * ATTENTION_HEAD_DIM
CONDITIONING_CHANNELS = 3


class ModelVariant(StrEnum):
    """Available SD3.5 ControlNet ComfyUI model variants."""

    SD35_CONTROLNET_BLUR = "ControlNet_Blur"
    SD35_CONTROLNET_CANNY = "ControlNet_Canny"
    SD35_CONTROLNET_DEPTH = "ControlNet_Depth"


class ModelLoader(ForgeModel):
    """SD3.5 ControlNets ComfyUI Repackaged model loader."""

    _VARIANTS = {
        ModelVariant.SD35_CONTROLNET_BLUR: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.SD35_CONTROLNET_CANNY: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.SD35_CONTROLNET_DEPTH: ModelConfig(
            pretrained_model_name=REPO_ID,
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
            _CONFIGS[version],
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

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the SD3.5 ControlNet.

        Returns a dict matching SD3ControlNetModel.forward() signature.
        These controlnets have no pos_embed or context_embedder, so
        hidden_states must be 3D (pre-patchified) and encoder_hidden_states
        must not be provided.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        seq_len = (SAMPLE_SIZE // PATCH_SIZE) ** 2
        hidden_states = torch.randn(batch_size, seq_len, INNER_DIM, dtype=dtype)
        pooled_projections = torch.randn(batch_size, 2048, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        controlnet_cond = torch.randn(
            batch_size, CONDITIONING_CHANNELS, SAMPLE_SIZE, SAMPLE_SIZE, dtype=dtype
        )

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "conditioning_scale": 1.0,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
        }
