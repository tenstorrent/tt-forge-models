# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MapAnything v1 model loader implementation for metric 3D reconstruction.

Loads the MapAnything model which performs universal feed-forward metric 3D
reconstruction from images, supporting tasks like depth estimation, multi-view
stereo, and structure from motion.

Requires the map-anything repository to be cloned at /tmp/map_anything_repo.
"""
import os
import sys
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

MAP_ANYTHING_REPO_PATH = "/tmp/map_anything_repo"


def _ensure_map_anything_importable():
    """Ensure the map-anything repo is cloned and importable."""
    if not os.path.isdir(MAP_ANYTHING_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/facebookresearch/map-anything.git",
                MAP_ANYTHING_REPO_PATH,
            ]
        )
        subprocess.check_call(
            ["pip", "install", "-e", MAP_ANYTHING_REPO_PATH],
        )
    if MAP_ANYTHING_REPO_PATH not in sys.path:
        sys.path.insert(0, MAP_ANYTHING_REPO_PATH)


class ModelVariant(StrEnum):
    """Available MapAnything v1 model variants."""

    DEFAULT = "Default"


class MapAnythingWrapper(torch.nn.Module):
    """Wrapper around MapAnything model for standard forward pass interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # pixel_values is [B, 3, H, W], already DINOv2-normalized
        views = [{"img": pixel_values, "data_norm_type": ["dinov2"]}]
        predictions = self.model.infer(views)
        return predictions[0]["pts3d"]


class ModelLoader(ForgeModel):
    """MapAnything v1 model loader for metric 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/map-anything-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MapAnything-v1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MapAnything v1 model.

        Returns:
            torch.nn.Module: The MapAnything model wrapped for standard inference.
        """
        _ensure_map_anything_importable()
        from mapanything.models import MapAnything

        repo_id = self._variant_config.pretrained_model_name
        model = MapAnything.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return MapAnythingWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the MapAnything v1 model.

        Returns:
            torch.Tensor: A batch of images as pixel values [B, 3, H, W].
        """
        from io import BytesIO

        import requests
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, ToTensor

        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
        try:
            response = requests.get(url, timeout=30)
            image = (
                Image.open(BytesIO(response.content)).convert("RGB").resize((518, 518))
            )
        except Exception:
            image = Image.new("RGB", (518, 518), color=(128, 128, 128))

        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            ]
        )
        pixel_values = transform(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
