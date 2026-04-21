# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V3 (DA3) model loader implementation for monocular depth estimation.
"""
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
    """Wrapper around the inner DA3 network (DepthAnything3Net) that takes a
    preprocessed image tensor and returns depth prediction."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # Input is (B, 3, H, W), model expects (B, N, 3, H, W)
        x = pixel_values.unsqueeze(1).float()
        output = self.model(x)
        return output.depth


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
        import sys

        # The local "evo/" model directory shadows the pip "evo" package
        # required by depth_anything_3. Temporarily fix the import resolution.
        stale_evo = sys.modules.pop("evo", None)
        repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        path_modified = False
        if repo_root in sys.path:
            sys.path.remove(repo_root)
            path_modified = True
        try:
            from depth_anything_3.api import DepthAnything3
        finally:
            if path_modified:
                sys.path.insert(0, repo_root)
            if stale_evo is not None:
                sys.modules["evo"] = stale_evo

        pretrained_model_name = self._variant_config.pretrained_model_name

        api_model = DepthAnything3.from_pretrained(pretrained_model_name)
        api_model.eval()

        wrapper = DepthAnything3Wrapper(api_model.model)
        wrapper.eval()

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"].convert("RGB")

        image_np = np.array(image)
        pixel_values = (
            torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
