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


class DepthAnything3Wrapper(nn.Module):
    """Wrapper around the inner DepthAnything3Net that takes a preprocessed
    image tensor of shape (B, N, 3, H, W) and returns the depth map."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, pixel_values):
        # Keep in float32; internal autocast is skipped at this level
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
        # The local 'evo/' and 'spacy/' model directories in tt_forge_models
        # shadow the pip packages required by depth_anything_3. Temporarily
        # remove tt_forge_models roots from sys.path and stash cached namespace
        # packages so the real pip packages are found.
        original_path, stashed = self._clear_shadow_modules()
        try:
            from depth_anything_3.api import DepthAnything3
            from depth_anything_3.model.dinov2.layers.rope import PositionGetter
        finally:
            self._restore_shadow_modules(original_path, stashed)

        # PositionGetter caches positions keyed only by (height, width), not by
        # device. If the cache was populated on CPU (e.g. during model init),
        # subsequent XLA calls return CPU tensors, breaking torch.cat with XLA
        # tensors during Dynamo tracing. Patch __call__ to move cached positions
        # to the requested device before returning.
        _orig_position_getter_call = PositionGetter.__call__

        def _device_aware_call(self_pg, batch_size, height, width, device):
            positions = _orig_position_getter_call(self_pg, batch_size, height, width, device)
            return positions.to(device)

        PositionGetter.__call__ = _device_aware_call

        pretrained_model_name = self._variant_config.pretrained_model_name

        da3 = DepthAnything3.from_pretrained(pretrained_model_name)
        da3.eval()

        wrapper = DepthAnything3Wrapper(da3.model)
        wrapper.eval()

        return wrapper

    def _clear_shadow_modules(self):
        """Remove tt_forge_models roots from sys.path and evict namespace-package
        shadows for evo and spacy (local model directories that shadow pip pkgs).

        Evo shadows are stashed and restored after the import so other callers
        still see the real evo module. Spacy is NOT a real installed package here,
        so the namespace-package proxy is dropped permanently — this prevents
        datasets._dill from hitting AttributeError on spacy.Language.
        """
        loader_dir = os.path.dirname(os.path.abspath(__file__))
        models_root = os.path.dirname(os.path.dirname(loader_dir))
        shadow_dirs = {models_root, os.getcwd(), ""}
        original_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in shadow_dirs]
        # Drop evo namespace proxy so depth_anything_3 finds the real pip evo.
        stashed_evo = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "evo" or k.startswith("evo.")
        }
        # Drop spacy namespace proxy permanently; it has no Language attribute
        # and causes datasets._dill to raise AttributeError.
        for k in list(sys.modules):
            if k == "spacy" or k.startswith("spacy."):
                del sys.modules[k]
        return original_path, stashed_evo

    def _restore_shadow_modules(self, original_path, stashed_evo):
        sys.path = original_path
        for k, v in stashed_evo.items():
            sys.modules.setdefault(k, v)

    def load_inputs(self, dtype_override=None, batch_size=1):
        original_path, stashed = self._clear_shadow_modules()
        try:
            from datasets import load_dataset
        finally:
            self._restore_shadow_modules(original_path, stashed)
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
