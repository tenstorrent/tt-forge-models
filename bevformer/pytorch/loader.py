# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Detr3d model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from .src.model import BEVFormer, img_backbone, pts_bbox_head, img_neck
from .src.checkpoint import load_checkpoint
from .src.nuscenes_dataloader import build_dataloader
from .src.nuscenes_dataset import build_dataset, data_test
from loguru import logger


class ModelVariant(StrEnum):
    """Available BEVFormer model variants."""

    BEVFORMER_TINY = "BEVFormer-tiny"


class ModelLoader(ForgeModel):
    """BEVFormer model loader implementation for autonomous driving tasks."""

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

    def load_model(self, **kwargs):
        """Load and return the BEVFormer model instance with default settings.
        Returns:
            Torch model: The BEVFormer model instance.
        """
        # Load model with defaults
        model = BEVFormer(
            img_backbone=img_backbone,
            pts_bbox_head=pts_bbox_head,
            img_neck=img_neck,
            use_grid_mask=True,
            video_test_mode=True,
        )
        checkpoint = load_checkpoint(
            model,
            "/proj_sw/user_dev/mramanathan/bgdlab19_sep10_xla/tt-xla/third_party/tt_forge_models/bevformer/pytorch/src/bevformer_tiny_epoch_24.pth",
            map_location="cpu",
        )
        model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the BEVFormer model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        dataset = build_dataset(data_test)
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
