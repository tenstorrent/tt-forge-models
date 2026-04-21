# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Augmented CAMUS Segmentation ONNX model loader implementation for cardiac
ultrasound segmentation (left ventricle and myocardium).
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
    """Available Augmented CAMUS Segmentation ONNX model variants."""

    AUGMENTED_CAMUS_SEGMENTATION = "augmented_camus_segmentation"


class ModelLoader(ForgeModel):
    """Augmented CAMUS Segmentation ONNX model loader for cardiac ultrasound."""

    _VARIANTS = {
        ModelVariant.AUGMENTED_CAMUS_SEGMENTATION: ModelConfig(
            pretrained_model_name="zeahub/augmented-camus-segmentation",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AUGMENTED_CAMUS_SEGMENTATION

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Augmented CAMUS Segmentation",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Augmented CAMUS Segmentation ONNX model.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Augmented CAMUS Segmentation model.

        The model expects grayscale cardiac ultrasound images of the apical
        two-chamber or four-chamber view. Input shape: [1, 1, 256, 256].

        Returns:
            torch.Tensor: Sample input tensor of shape [1, 1, 256, 256].
        """
        inputs = torch.rand(1, 1, 256, 256)

        return inputs
