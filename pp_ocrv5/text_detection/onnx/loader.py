# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PP-OCRv5 Chinese Mobile Detection ONNX model loader implementation.
"""

from typing import Optional

import onnx
import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PP-OCRv5 text detection model variants (ONNX)."""

    CH_PP_OCRV5_DET = "ch_PP-OCRv5_det"


class ModelLoader(ForgeModel):
    """PP-OCRv5 Chinese Mobile Detection ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.CH_PP_OCRV5_DET: ModelConfig(
            pretrained_model_name="breezedeus/cnstd-ppocr-ch_PP-OCRv5_det",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CH_PP_OCRV5_DET

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PP-OCRv5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load pretrained PP-OCRv5 Chinese mobile detection model (ONNX)."""
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(pretrained, "ch_PP-OCRv5_det_infer.onnx")
        return onnx.load(model_path)

    def load_inputs(self, **kwargs):
        """Prepare sample input for PP-OCRv5 Chinese mobile detection (ONNX).

        The detector accepts NCHW float32 images normalized with ImageNet
        statistics. Spatial dimensions must be multiples of 32.
        """
        return torch.randn(1, 3, 640, 640)
