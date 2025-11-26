# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MaskFormer Swin-B model loader implementation for segmentation tasks.
"""

import torch
from PIL import Image
from typing import Optional
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

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
from ...tools.utils import get_file
from .src.model_utils import MaskformerEncoderWrapper


class ModelVariant(StrEnum):
    """Available MaskFormer Swin-B model variants for segmentation."""

    SWIN_B_COCO = "swin_base_coco"
    SWIN_B_ADE = "swin_base_ade"


class ModelLoader(ForgeModel):
    """MaskFormer Swin-B model loader implementation for segmentation tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SWIN_B_COCO: ModelConfig(
            pretrained_model_name="facebook/maskformer-swin-base-coco"
        ),
        ModelVariant.SWIN_B_ADE: ModelConfig(
            pretrained_model_name="facebook/maskformer-swin-base-ade"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SWIN_B_COCO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="maskformer_swin_b",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        """Load feature extractor for the current variant.

        Returns:
            The loaded feature extractor instance
        """
        # Load the feature extractor
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.feature_extractor

    def load_model(self, dtype_override=None):
        """Load and return the MaskFormer model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).

        Returns:
            torch.nn.Module: The MaskFormer model instance for segmentation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # NOTE: Ignoring dtype_override and always using default (fp32) due to dtype mismatch
        # issue with bfloat16. See: https://github.com/tenstorrent/tt-xla/issues/1959
        # if dtype_override is not None:
        #     model_kwargs["torch_dtype"] = dtype_override

        model = MaskFormerForInstanceSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model = model.model.pixel_level_module.encoder.model.encoder
        model = MaskformerEncoderWrapper(model, num_layers=2)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the MaskFormer model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           NOTE: This parameter is currently ignored (inputs always use float32).
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        hidden_states = torch.load(
            "third_party/tt_forge_models/maskformer_swin_b/pytorch/src/hidden_states.pt"
        )

        return hidden_states
