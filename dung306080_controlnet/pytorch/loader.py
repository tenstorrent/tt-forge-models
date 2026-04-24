# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dung306080/Controlnet model loader implementation.

Loads single-file SDXL ControlNet safetensors checkpoints mirrored in the
Dung306080/Controlnet HuggingFace repo. Upstream diffusers configs are used
for model construction since the repo ships raw weights without config.json.

Available variants:
- XINSIR_CANNY_SDXL: xinsir ControlNet Canny for SDXL
- XINSIR_DEPTH_SDXL: xinsir ControlNet Depth for SDXL
- XINSIR_OPENPOSE_SDXL: xinsir ControlNet OpenPose for SDXL
- XINSIR_TILE_SDXL: xinsir ControlNet Tile for SDXL
- THIBAUD_OPENPOSE_XL2: thibaud OpenPoseXL2 ControlNet for SDXL
"""

from typing import Any, Optional

import torch
from diffusers import ControlNetModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "Dung306080/Controlnet"

# ControlNet safetensors file names within the Dung306080/Controlnet repo.
_CONTROLNET_FILES = {
    "xinsir_canny": "xinsir-controlnet-canny-sdxl.safetensors",
    "xinsir_depth": "xinsir-controlnet-depth-sdxl-1.0.safetensors",
    "xinsir_openpose": "xinsir-controlnet-openpose-sdxl-1.0.safetensors",
    "xinsir_tile": "xinsir-controlnet-tile-sdxl-1.0.safetensors",
    "thibaud_openpose_xl2": "OpenPoseXL2.safetensors",
}

# Upstream diffusers config sources for each variant.
_CONFIGS = {
    "xinsir_canny": "xinsir/controlnet-canny-sdxl-1.0",
    "xinsir_depth": "xinsir/controlnet-depth-sdxl-1.0",
    "xinsir_openpose": "xinsir/controlnet-openpose-sdxl-1.0",
    "xinsir_tile": "xinsir/controlnet-tile-sdxl-1.0",
    "thibaud_openpose_xl2": "thibaud/controlnet-openpose-sdxl-1.0",
}

# SDXL ControlNet dimensions (derived from SDXL base UNet).
SAMPLE_SIZE = 128
LATENT_CHANNELS = 4
CROSS_ATTENTION_DIM = 2048
CONDITIONING_CHANNELS = 3
PROJECTION_DIM = 1280
ADD_TIME_IDS_DIM = 6
IMAGE_SIZE = 1024


class ModelVariant(StrEnum):
    """Available Dung306080/Controlnet variants."""

    XINSIR_CANNY_SDXL = "xinsir_canny_sdxl"
    XINSIR_DEPTH_SDXL = "xinsir_depth_sdxl"
    XINSIR_OPENPOSE_SDXL = "xinsir_openpose_sdxl"
    XINSIR_TILE_SDXL = "xinsir_tile_sdxl"
    THIBAUD_OPENPOSE_XL2 = "thibaud_openpose_xl2"


class ModelLoader(ForgeModel):
    """Dung306080/Controlnet model loader implementation."""

    _VARIANTS = {
        ModelVariant.XINSIR_CANNY_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.XINSIR_DEPTH_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.XINSIR_OPENPOSE_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.XINSIR_TILE_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.THIBAUD_OPENPOSE_XL2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.XINSIR_CANNY_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Dung306080_Controlnet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        """Map variant to version key for file/config lookup."""
        return {
            ModelVariant.XINSIR_CANNY_SDXL: "xinsir_canny",
            ModelVariant.XINSIR_DEPTH_SDXL: "xinsir_depth",
            ModelVariant.XINSIR_OPENPOSE_SDXL: "xinsir_openpose",
            ModelVariant.XINSIR_TILE_SDXL: "xinsir_tile",
            ModelVariant.THIBAUD_OPENPOSE_XL2: "thibaud_openpose_xl2",
        }[self._variant]

    def _load_controlnet(self, dtype: torch.dtype = torch.float32) -> ControlNetModel:
        """Load ControlNet from single-file safetensors."""
        version = self._get_version_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_CONTROLNET_FILES[version],
        )

        self._controlnet = ControlNetModel.from_single_file(
            model_path,
            config=_CONFIGS[version],
            torch_dtype=dtype,
        )
        self._controlnet = self._controlnet.to(dtype=dtype)
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Dung306080/Controlnet model.

        Returns:
            ControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Any:
        """Prepare sample inputs for the SDXL ControlNet.

        Returns a dict matching ControlNetModel.forward() signature for SDXL.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        batch_size = kwargs.get("batch_size", 1)

        latent_h = IMAGE_SIZE // 8
        latent_w = IMAGE_SIZE // 8

        sample = torch.randn(
            batch_size, LATENT_CHANNELS, latent_h, latent_w, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, 77, CROSS_ATTENTION_DIM, dtype=dtype
        )
        controlnet_cond = torch.randn(
            batch_size, CONDITIONING_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype
        )
        added_cond_kwargs = {
            "text_embeds": torch.randn(batch_size, PROJECTION_DIM, dtype=dtype),
            "time_ids": torch.randn(batch_size, ADD_TIME_IDS_DIM, dtype=dtype),
        }

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "added_cond_kwargs": added_cond_kwargs,
        }
