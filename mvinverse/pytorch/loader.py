# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MVInverse model loader implementation for multi-view inverse rendering.

MVInverse is a feed-forward framework that decomposes multi-view images into
physically-based material components (albedo, metallic, roughness, normals,
shading) using alternating intra-view / inter-view attention over a DINOv2
ViT-L/14 backbone.

The weights on the Hugging Face Hub are published via ``PyTorchModelHubMixin``,
so the underlying :class:`MVInverse` class is required to materialize the model.
That class lives in the upstream project, which is cloned on demand to
``/tmp/mvinverse_repo``.
"""

import os
import sys
import types
from typing import Optional

import torch

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

REPO_ID = "maddog241/mvinverse"
MVINVERSE_REPO_PATH = "/tmp/mvinverse_repo"


def _ensure_mvinverse_importable():
    """Ensure the upstream mvinverse repo is cloned and importable.

    The upstream project ships its model code under a top-level ``mvinverse``
    package, which collides with this loader's own ``mvinverse`` directory in
    tt-forge-models. Bind ``sys.modules['mvinverse']`` to the cloned package
    so ``from mvinverse.models.mvinverse import MVInverse`` resolves to the
    upstream sources regardless of sys.path ordering.
    """
    upstream_pkg = sys.modules.get("mvinverse")
    upstream_path = os.path.join(MVINVERSE_REPO_PATH, "mvinverse")
    if (
        upstream_pkg is not None
        and getattr(upstream_pkg, "__path__", None)
        and upstream_path in list(upstream_pkg.__path__)
    ):
        return

    if not os.path.isdir(MVINVERSE_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/Maddog241/mvinverse.git",
                MVINVERSE_REPO_PATH,
            ]
        )

    if MVINVERSE_REPO_PATH not in sys.path:
        sys.path.insert(0, MVINVERSE_REPO_PATH)

    pkg = types.ModuleType("mvinverse")
    pkg.__path__ = [upstream_path]
    sys.modules["mvinverse"] = pkg


class ModelVariant(StrEnum):
    """Available MVInverse model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MVInverse model loader for multi-view inverse rendering."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    # DINOv2 ViT-L/14 backbone requires H and W divisible by 14.
    _NUM_VIEWS = 4
    _IMAGE_SIZE = 224

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MVInverse",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MVInverse model from the Hugging Face Hub.

        Returns:
            torch.nn.Module: The MVInverse multi-view inverse rendering model.
        """
        _ensure_mvinverse_importable()
        from mvinverse.models.mvinverse import MVInverse

        repo_id = self._variant_config.pretrained_model_name
        model = MVInverse.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample multi-view image inputs for MVInverse.

        Returns:
            torch.Tensor: Multi-view images of shape [B, N, 3, H, W].
        """
        dtype = dtype_override or torch.float32

        images = torch.randn(
            batch_size,
            self._NUM_VIEWS,
            3,
            self._IMAGE_SIZE,
            self._IMAGE_SIZE,
            dtype=dtype,
        )

        return images
