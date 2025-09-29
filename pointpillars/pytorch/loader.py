# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PointPillars model loader implementation
"""
import os
import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src import PointPillarsCore, PointPillarsPre, PointPillarsPos
from .src.utils import read_points, point_range_filter, keep_bbox_from_lidar_range
from ...tools.utils import get_file


@dataclass
class PConfig(ModelConfig):
    source: ModelSource = ModelSource.CUSTOM


class ModelVariant(StrEnum):
    CORE = "pointpillars"


class ModelLoader(ForgeModel):
    """PointPillars model loader implementation."""

    _VARIANTS = {
        ModelVariant.CORE: PConfig(pretrained_model_name="pointpillars"),
    }

    DEFAULT_VARIANT = ModelVariant.CORE

    # PointPillars constants
    CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    PCD_LIMIT_RANGE = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="pointpillars",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the PointPillars model instance for this instance's variant."""
        checkpoint = get_file(
            "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pretrained/epoch_160.pth"
        )
        model = PointPillarsCore(nclasses=len(self.CLASSES))
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        return model

    def load_inputs(self):
        """Load and return sample inputs for the PointPillars model."""
        pre_processor_layer = PointPillarsPre()
        pre_processor_layer.eval()

        pc_path = get_file(
            "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pointpillars/dataset/demo_data/val/000134.bin"
        )
        # Load actual point cloud data
        pc = read_points(pc_path)
        # Apply point cloud filtering
        pc = point_range_filter(pc, self.PCD_LIMIT_RANGE)
        # Convert to torch tensor
        pc_torch = torch.from_numpy(pc)

        with torch.no_grad():
            pillars, coors_batch, npoints_per_pillar = pre_processor_layer(
                batched_pts=[pc_torch]
            )

        return (pillars, coors_batch, npoints_per_pillar)

    def post_processing(self, co_out):
        """Post-process the PointPillars model outputs."""
        model_post = PointPillarsPos(nclasses=len(self.CLASSES))
        model_post.eval()

        with torch.no_grad():
            result_filter = model_post(co_out)
        result_filter = keep_bbox_from_lidar_range(
            result_filter[0], self.PCD_LIMIT_RANGE
        )
        lidar_bboxes = result_filter["lidar_bboxes"]
        labels = result_filter["labels"]
        scores = result_filter["scores"]

        # Convert labels to class names
        class_names = [self.LABEL2CLASSES[label] for label in labels]

        print("Objects Found in Pointclouds are:", class_names)
