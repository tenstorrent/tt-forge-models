# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PAN (Pixel Attention Network) model loader implementation for image super-resolution.

Loads the eugenesiow/pan pretrained weights at 2x, 3x, or 4x upscale factors.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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
from .src.pan_model import PAN


@dataclass
class PanConfig(ModelConfig):
    """PAN model configuration with scale factor and weights filename."""

    scale: int = 2
    weights_filename: str = "pytorch_model_2x.pt"


class ModelVariant(StrEnum):
    """Available PAN model variants."""

    SCALE_2X = "scale_2x"
    SCALE_3X = "scale_3x"
    SCALE_4X = "scale_4x"


class ModelLoader(ForgeModel):
    """PAN model loader for image super-resolution."""

    _VARIANTS = {
        ModelVariant.SCALE_2X: PanConfig(
            pretrained_model_name="eugenesiow/pan",
            scale=2,
            weights_filename="pytorch_model_2x.pt",
        ),
        ModelVariant.SCALE_3X: PanConfig(
            pretrained_model_name="eugenesiow/pan",
            scale=3,
            weights_filename="pytorch_model_3x.pt",
        ),
        ModelVariant.SCALE_4X: PanConfig(
            pretrained_model_name="eugenesiow/pan",
            scale=4,
            weights_filename="pytorch_model_4x.pt",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SCALE_2X

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PAN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the PAN super-resolution model."""
        cfg = self._variant_config
        model = PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=cfg.scale)

        weights_path = hf_hub_download(
            repo_id=cfg.pretrained_model_name,
            filename=cfg.weights_filename,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, batch_size=1):
        """Load and return sample inputs for the PAN model."""
        inputs = torch.randn(batch_size, 3, 64, 64)
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return inputs
