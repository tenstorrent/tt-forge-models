# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Merlin model loader implementation.

Loads stanfordmimi/Merlin, a 3D vision-language foundation model for computed
tomography (CT) imaging. Merlin pairs an inflated ResNet-152 (I3ResNet) vision
encoder with a Clinical Longformer text encoder. This loader exposes the
image-only branch (ImageEmbedding=True), which produces 2048-dim image
embeddings suitable for downstream tasks.
"""

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

REPO_ID = "stanfordmimi/Merlin"

# Tensor layout matches the official monai preprocessing: (B, C, H, W, D).
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_DEPTH = 160


class ModelVariant(StrEnum):
    """Available Merlin model variants."""

    IMAGE_EMBEDDING = "image_embedding"


class ModelLoader(ForgeModel):
    """Merlin CT vision-language model loader (image-embedding branch)."""

    _VARIANTS = {
        ModelVariant.IMAGE_EMBEDDING: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.IMAGE_EMBEDDING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Merlin",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Merlin image-embedding model.

        Returns:
            The Merlin nn.Module configured for image-only embedding output.
        """
        from merlin import Merlin  # type: ignore[import]

        if self._model is None:
            self._model = Merlin(ImageEmbedding=True)
            self._model.eval()
            if dtype_override is not None:
                self._model = self._model.to(dtype=dtype_override)
        elif dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Prepare a synthetic 3D CT volume for Merlin.

        Returns:
            Image tensor of shape (1, 1, H, W, D) matching the preprocessing
            pipeline used in the official demo.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        return torch.randn(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, dtype=dtype)
