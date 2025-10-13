# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormer model loader implementation
"""
from typing import Optional
from ...base import ForgeModel
from ...config import (
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelInfo,
    ModelConfig,
)
from .src.model import BEVFormer, get_bevformer_model
from .src.checkpoint import load_checkpoint
from .src.nuscenes_dataloader import build_dataloader
from .src.nuscenes_dataset import build_dataset, get_test_dataset_cfg
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available BEVFormer model variants."""

    BEVFORMER_TINY = "BEVFormer-tiny"
    BEVFORMER_SMALL = "BEVFormer-small"
    BEVFORMER_BASE = "BEVFormer-base"


class ModelLoader(ForgeModel):
    """BEVFormer model loader implementation for autonomous driving tasks."""

    _VARIANTS = {
        ModelVariant.BEVFORMER_TINY: ModelConfig(
            pretrained_model_name="BEVFormer-tiny"
        ),
        ModelVariant.BEVFORMER_SMALL: ModelConfig(
            pretrained_model_name="BEVFormer-small"
        ),
        ModelVariant.BEVFORMER_BASE: ModelConfig(
            pretrained_model_name="BEVFormer-base"
        ),
    }
    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEVFORMER_TINY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
        self.processor = None

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
            model="bevformer",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Load and return the BEVFormer model instance with default settings.
        Returns:
            Torch model: The BEVFormer model instance.
        """
        variant_str = str(self._variant) if self._variant else str(self.DEFAULT_VARIANT)
        img_backbone, pts_bbox_head, img_neck = get_bevformer_model(variant_str)
        model = BEVFormer(
            img_backbone=img_backbone,
            pts_bbox_head=pts_bbox_head,
            img_neck=img_neck,
            use_grid_mask=True,
            video_test_mode=True,
        )
        if variant_str == ModelVariant.BEVFORMER_SMALL.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_small_epoch_24.pth")
            )
        elif variant_str == ModelVariant.BEVFORMER_BASE.value:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_r101_dcn_24ep.pth")
            )
        else:
            checkpoint_path = str(
                get_file("test_files/pytorch/bevformer/bevformer_tiny_epoch_24.pth")
            )
        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
            map_location="cpu",
        )
        model.eval()
        return model

    def load_inputs(self, variant: Optional["ModelVariant"] = None, **kwargs):
        """Return sample inputs for the BEVFormer model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        variant_str = str(self._variant) if self._variant else str(self.DEFAULT_VARIANT)
        dataset = build_dataset(get_test_dataset_cfg(variant_str))

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=4,
            dist=True,
            shuffle=False,
            nonshuffler_sampler={"type": "DistributedSampler"},
        )
        for k in data_loader:
            input_image = k
            break

        return input_image
