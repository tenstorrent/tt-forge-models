# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Augmented CAMUS segmentation ONNX model loader.

Loads the zeahub/augmented-camus-segmentation nnU-Net model for cardiac
ultrasound segmentation of the left ventricle and myocardium on apical
two-chamber and four-chamber views.

Reference: Van De Vyver, Gilles, et al. "Generative augmentations for
improved cardiac ultrasound segmentation using diffusion models."
arXiv:2502.20100 (2025).
"""

from typing import Optional

import onnx
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


class ModelVariant(StrEnum):
    """Available Augmented CAMUS segmentation variants."""

    AUGMENTED_CAMUS_SEG = "augmented_camus_seg"


class ModelLoader(ForgeModel):
    """Augmented CAMUS ONNX loader for cardiac ultrasound segmentation."""

    _VARIANTS = {
        ModelVariant.AUGMENTED_CAMUS_SEG: ModelConfig(
            pretrained_model_name="zeahub/augmented-camus-segmentation",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AUGMENTED_CAMUS_SEG

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AugmentedCamusSeg",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Augmented CAMUS ONNX model."""
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.onnx",
        )
        return onnx.load(model_path)

    def load_inputs(self, **kwargs):
        """Return a sample grayscale cardiac ultrasound input tensor.

        The model expects a single-channel 256x256 image; intensities are
        normalized internally so no external pre-processing is required.
        """
        return torch.rand(1, 1, 256, 256)
