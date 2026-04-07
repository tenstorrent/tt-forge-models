# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSD512 model loader implementation
"""

from typing import Optional
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
import torch
from ....base import ForgeModel
from .src.model import build_ssd
from .src.model_utils import load_ssd512_inputs, postprocess_outputs
from ....tools.utils import get_file
from .src.model import build_ssd


class ModelVariant(StrEnum):
    """Available SSD512 model variants."""

    SSD512 = "ssd512"


class ModelLoader(ForgeModel):
    """SSD512 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SSD512: ModelConfig(
            pretrained_model_name="ssd512",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SSD512

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.num_classes = 21

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ssd512",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SSD512 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).

        Returns:
            torch.nn.Module: The SSD512 model instance.
        """
        model = build_ssd("test", 300, self.num_classes)
        weights = get_file(str("test_files/pytorch/ssd512/ssd300_mAP_77.43_v2.pth"))
        state_dict = torch.load(weights, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SSD512 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for SSD512.
        """
        return load_ssd512_inputs()

    def postprocess_outputs(self, outputs):
        """Postprocess the model outputs to return the bounding boxes and scores.

        Args:
            outputs: The model outputs to postprocess.

        """
        return postprocess_outputs(outputs)
