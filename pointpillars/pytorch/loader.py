# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PointPillars model loader implementation
"""
import os
import numpy as np
import torch
import torch.nn as nn
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
from .src import (
    PointPillarsCore,
    PointPillarsPre,
    PointPillarsPos,
    get_predicted_bboxes,
)
from .src.utils import read_points, point_range_filter, keep_bbox_from_lidar_range
from ...tools.utils import get_file


@dataclass
class PConfig(ModelConfig):
    source: ModelSource = ModelSource.CUSTOM


class ModelVariant(StrEnum):
    CORE = "pointpillars"
    # BEV_INPUT: pre-scatter on CPU → pass static BEV map (1,64,496,432) to TT.
    # Avoids the ttir.gather that the pillar-scatter produces.
    BEV_INPUT = "pointpillars_bev"


class PointPillarsBEVCore(nn.Module):
    """PointPillars backbone + neck + head that accepts a pre-scattered BEV map.

    The pillar encoder (including the scatter that produces ttir.gather) is run
    entirely on CPU in load_inputs().  This module only contains the learned
    2D-conv backbone, FPN neck, and detection head — all ops TT supports.

    Input : bev_map  (1, 64, 496, 432)  bfloat16
    Output: (bbox_cls_pred, bbox_pred, bbox_dir_cls_pred)
              (1,18,248,216)  (1,42,248,216)  (1,12,248,216)
    """

    def __init__(self, nclasses: int = 3):
        super().__init__()
        core = PointPillarsCore(nclasses=nclasses)
        self.backbone = core.backbone
        self.neck = core.neck
        self.head = core.head

    def load_weights_from_core_state_dict(self, state_dict: dict):
        own = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(own)
        own = {k[len("neck."):]: v for k, v in state_dict.items() if k.startswith("neck.")}
        self.neck.load_state_dict(own)
        own = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
        self.head.load_state_dict(own)

    def forward(self, bev_map):
        x = self.backbone(bev_map)
        x = self.neck(x)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class ModelLoader(ForgeModel):
    """PointPillars model loader implementation."""

    _VARIANTS = {
        ModelVariant.CORE: PConfig(pretrained_model_name="pointpillars"),
        ModelVariant.BEV_INPUT: PConfig(pretrained_model_name="pointpillars"),
    }

    DEFAULT_VARIANT = ModelVariant.BEV_INPUT

    # PointPillars constants
    CLASSES = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    PCD_LIMIT_RANGE = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PointPillars",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _load_checkpoint(self):
        return get_file(
            "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pretrained/epoch_160.pth"
        )

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the PointPillars model instance for this instance's variant."""
        checkpoint = self._load_checkpoint()

        if self._variant == ModelVariant.BEV_INPUT:
            model = PointPillarsBEVCore(nclasses=len(self.CLASSES))
            state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
            model.load_weights_from_core_state_dict(state_dict)
            model.eval()
            if dtype_override is not None:
                model = model.to(dtype_override)
            self.model = model
            return model

        # CORE variant (original behaviour)
        model = PointPillarsCore(nclasses=len(self.CLASSES))
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the PointPillars model."""
        if self._variant == ModelVariant.BEV_INPUT:
            return self._load_bev_inputs(dtype_override)
        return self._load_core_inputs()

    def _load_bev_inputs(self, dtype_override=None):
        """Pre-compute BEV map on CPU — the static-shape input for TT."""
        checkpoint = self._load_checkpoint()
        full_core = PointPillarsCore(nclasses=len(self.CLASSES))
        full_core.load_state_dict(
            torch.load(checkpoint, map_location=torch.device("cpu"))
        )
        full_core.eval()

        pc_path = get_file(
            "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pointpillars/dataset/demo_data/val/000134.bin"
        )
        pc = read_points(pc_path)
        pc = point_range_filter(pc, self.PCD_LIMIT_RANGE)
        pc_torch = torch.from_numpy(pc)

        pre = PointPillarsPre()
        pre.eval()
        with torch.no_grad():
            pillars, coors_batch, npoints = pre([pc_torch])
            bev_map = full_core.pillar_encoder(pillars, coors_batch, npoints)

        if dtype_override is not None:
            bev_map = bev_map.to(dtype_override)
        return (bev_map,)  # (1, 64, 496, 432) — fully static

    def _load_core_inputs(self):
        """Original CORE variant inputs."""
        pre_processor_layer = PointPillarsPre()
        pre_processor_layer.eval()

        pc_path = get_file(
            "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pointpillars/dataset/demo_data/val/000134.bin"
        )
        pc = read_points(pc_path)
        pc = point_range_filter(pc, self.PCD_LIMIT_RANGE)
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

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors = co_out
        results = get_predicted_bboxes(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_anchors=batched_anchors,
            nclasses=len(self.CLASSES),
            nms_pre=self.model.nms_pre,
        )

        with torch.no_grad():
            result_filter = model_post(results)
        result_filter = keep_bbox_from_lidar_range(
            result_filter[0], self.PCD_LIMIT_RANGE
        )
        lidar_bboxes = result_filter["lidar_bboxes"]
        labels = result_filter["labels"]
        scores = result_filter["scores"]

        class_names = [self.LABEL2CLASSES[label] for label in labels]
        print("Objects Found in Pointclouds are:", class_names)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor."""
        if isinstance(fwd_output, tuple):
            tensors = [t for t in fwd_output if isinstance(t, torch.Tensor)]
            flattened = [t.flatten(start_dim=1) for t in tensors]
            return torch.cat(flattened, dim=1)
        elif isinstance(fwd_output, list):
            if len(fwd_output) > 0 and isinstance(fwd_output[0], torch.Tensor):
                flattened = [t.flatten() for t in fwd_output]
                return torch.cat(flattened, dim=0)
            return fwd_output[0] if fwd_output else fwd_output
        return fwd_output
