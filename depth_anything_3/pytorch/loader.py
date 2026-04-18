# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything 3 model loader implementation for monocular depth estimation.
"""
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Optional

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


class DA3Wrapper(nn.Module):
    """Wrapper around DA3 model for depth estimation inference."""

    def __init__(self, da3_api):
        super().__init__()
        self.model = da3_api.model

    def forward(self, pixel_values):
        # Model expects (B, N, C, H, W); add N=1 view dimension
        x = pixel_values.unsqueeze(1)
        output = self.model(x)
        return output["depth"]


class ModelVariant(StrEnum):
    """Available Depth Anything 3 model variants."""

    SMALL = "Small"
    GIANT = "Giant"


class ModelLoader(ForgeModel):
    """Depth Anything 3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="depth-anything/da3-small",
        ),
        ModelVariant.GIANT: ModelConfig(
            pretrained_model_name="depth-anything/DA3-GIANT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DepthAnything3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import sys
        import types

        # The pip-installed 'evo' package is shadowed by the local 'evo/' model
        # directory. Mock evo.core since depth_anything_3.api imports pose_align
        # which requires evo.core.trajectory (unused for inference).
        evo_core = types.ModuleType("evo.core")
        evo_core_trajectory = types.ModuleType("evo.core.trajectory")
        evo_core_trajectory.PosePath3D = type("PosePath3D", (), {})
        evo_core.trajectory = evo_core_trajectory
        sys.modules.setdefault("evo.core", evo_core)
        sys.modules.setdefault("evo.core.trajectory", evo_core_trajectory)

        from depth_anything_3.api import DepthAnything3
        from .src.model_utils import patch_da3_for_xla

        pretrained_model_name = self._variant_config.pretrained_model_name

        da3 = DepthAnything3.from_pretrained(pretrained_model_name)
        patch_da3_for_xla(da3.model)

        wrapper = DA3Wrapper(da3)
        wrapper.eval()

        return wrapper

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (518, 518))

        pixel_values = TF.to_tensor(image)
        pixel_values = TF.normalize(
            pixel_values,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        pixel_values = pixel_values.unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1)

        return pixel_values
