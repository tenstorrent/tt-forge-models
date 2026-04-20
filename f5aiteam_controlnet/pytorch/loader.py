# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
F5AITeam ControlNet model loader implementation.

Loads SDXL ControlNet variants from the f5aiteam/Controlnet aggregate
repository, which mirrors canonical diffusers-compatible single-file
ControlNet safetensors. Model construction uses upstream diffusers
configs for shape metadata, since the repo itself has no config.json.

Available variants:
- DIFFUSERS_XL_CANNY: SDXL ControlNet Canny
- DIFFUSERS_XL_DEPTH: SDXL ControlNet Depth

Repository: https://huggingface.co/f5aiteam/Controlnet
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

REPO_ID = "f5aiteam/Controlnet"

# Single-file safetensors within the aggregate repo
_CONTROLNET_FILES = {
    "canny": "diffusers_xl_canny_full.safetensors",
    "depth": "diffusers_xl_depth_full.safetensors",
}

# Upstream diffusers config sources for each variant
_CONFIGS = {
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}

# SDXL ControlNet input dimensions (kept modest for test memory footprint)
LATENT_SIZE = 64
IN_CHANNELS = 4
CROSS_ATTENTION_DIM = 2048
TEXT_EMBEDS_DIM = 1280
TIME_IDS_DIM = 6
CONDITIONING_CHANNELS = 3
VAE_SCALE_FACTOR = 8
SEQUENCE_LENGTH = 77


class ModelVariant(StrEnum):
    """Available F5AITeam ControlNet model variants."""

    DIFFUSERS_XL_CANNY = "diffusers_xl_canny"
    DIFFUSERS_XL_DEPTH = "diffusers_xl_depth"


class ModelLoader(ForgeModel):
    """F5AITeam ControlNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.DIFFUSERS_XL_CANNY: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DIFFUSERS_XL_DEPTH: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSERS_XL_CANNY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="F5AITeam ControlNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        """Map variant to version key for file/config lookup."""
        return {
            ModelVariant.DIFFUSERS_XL_CANNY: "canny",
            ModelVariant.DIFFUSERS_XL_DEPTH: "depth",
        }[self._variant]

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the SDXL ControlNet model.

        Args:
            dtype_override: Optional torch.dtype to override the default (float32).

        Returns:
            ControlNetModel: The loaded ControlNet instance in eval mode.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
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
        self._controlnet.eval()
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the SDXL ControlNet.

        Returns a dict matching ControlNetModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        sample = torch.randn(
            batch_size, IN_CHANNELS, LATENT_SIZE, LATENT_SIZE, dtype=dtype
        )
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, SEQUENCE_LENGTH, CROSS_ATTENTION_DIM, dtype=dtype
        )
        controlnet_cond = torch.randn(
            batch_size,
            CONDITIONING_CHANNELS,
            LATENT_SIZE * VAE_SCALE_FACTOR,
            LATENT_SIZE * VAE_SCALE_FACTOR,
            dtype=dtype,
        )
        text_embeds = torch.randn(batch_size, TEXT_EMBEDS_DIM, dtype=dtype)
        time_ids = torch.randn(batch_size, TIME_IDS_DIM, dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": controlnet_cond,
            "added_cond_kwargs": {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
        }
