# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V3 (DA3) model loader implementation for monocular depth estimation.
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class DepthAnything3Wrapper(nn.Module):
    """Wrapper around the inner DepthAnything3Net that takes a preprocessed
    image tensor of shape (B, N, 3, H, W) and returns the depth map."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        pixel_values = pixel_values.float()
        prediction = self.net(pixel_values)
        return prediction["depth"]


class ModelVariant(StrEnum):
    """Available Depth Anything V3 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Depth Anything V3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="depth-anything/DA3-BASE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DepthAnythingV3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # The local 'evo/' model directory (another model in tt_forge_models)
        # shadows the pip 'evo' package required by depth_anything_3.
        # Temporarily remove tt_forge_models roots from sys.path and clear
        # cached evo modules so the pip package is found.
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        models_root = os.path.dirname(os.path.dirname(loader_dir))
        shadow_dirs = {models_root, os.getcwd(), ""}
        original_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in shadow_dirs]
        stashed_evo = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "evo" or k.startswith("evo.")
        }
        try:
            from depth_anything_3.api import DepthAnything3
        finally:
            sys.path = original_path
            for k, v in stashed_evo.items():
                sys.modules.setdefault(k, v)

        pretrained_model_name = self._variant_config.pretrained_model_name

        da3 = DepthAnything3.from_pretrained(pretrained_model_name)
        da3.eval()

        wrapper = DepthAnything3Wrapper(da3.model)
        wrapper.eval()

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        from torchvision import transforms

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((504, 504)),
                transforms.ToTensor(),
            ]
        )

        # Model expects (B, N, 3, H, W) where N is the number of views
        pixel_values = transform(image).unsqueeze(0).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
