# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UFM (UniFlowMatch) model loader implementation for dense correspondence estimation
"""

import os
import subprocess
import sys
import torch
from typing import Optional

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

_UFM_REPO_URL = "https://github.com/UniFlowMatch/UFM.git"
_UFM_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "uniflowmatch", "UFM")


def _ensure_uniflowmatch_importable():
    try:
        from uniflowmatch.models.ufm import UniFlowMatchConfidence  # noqa: F401

        return
    except (ImportError, ModuleNotFoundError):
        pass

    if not os.path.exists(_UFM_CACHE_DIR):
        os.makedirs(os.path.dirname(_UFM_CACHE_DIR), exist_ok=True)
        subprocess.run(
            ["git", "clone", "--recurse-submodules", _UFM_REPO_URL, _UFM_CACHE_DIR],
            check=True,
        )

    uniception_path = os.path.join(_UFM_CACHE_DIR, "UniCeption")
    if _UFM_CACHE_DIR not in sys.path:
        sys.path.insert(0, _UFM_CACHE_DIR)
    if uniception_path not in sys.path:
        sys.path.insert(0, uniception_path)


class ModelVariant(StrEnum):
    """Available UFM model variants."""

    REFINE = "Refine"
    BASE = "Base"


class ModelLoader(ForgeModel):
    """UFM model loader implementation for dense correspondence estimation."""

    _VARIANTS = {
        ModelVariant.REFINE: ModelConfig(
            pretrained_model_name="infinity1096/UFM-Refine",
        ),
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="infinity1096/UFM-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REFINE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="UFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_uniflowmatch_importable()

        from uniflowmatch.models.ufm import (
            UniFlowMatchConfidence,
            UniFlowMatchClassificationRefinement,
        )

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._variant == ModelVariant.BASE:
            model = UniFlowMatchConfidence.from_pretrained(pretrained_model_name)
        else:
            model = UniFlowMatchClassificationRefinement.from_pretrained(
                pretrained_model_name
            )

        # UFM uses DINOv2 with layer norm that upcasts to float32, so dtype conversion
        # must not be applied — the model is designed to run with autocast instead.

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # UFM forward() expects view dicts with 'img' (B, 3, H, W) normalized tensors.
        # The model's inference resolution is 560x420 (W x H).
        # DINOv2 normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        height, width = 420, 560
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        raw1 = (
            torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32)
            / 255.0
        )
        raw2 = (
            torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32)
            / 255.0
        )
        img1 = (raw1 - mean) / std
        img2 = (raw2 - mean) / std

        view1 = {"img": img1, "symmetrized": False, "data_norm_type": "dinov2"}
        view2 = {"img": img2, "symmetrized": False, "data_norm_type": "dinov2"}

        return {"view1": view1, "view2": view2}
