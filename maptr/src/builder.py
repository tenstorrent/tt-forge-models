# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmdet.models.builder import DETECTORS
from mmcv.utils import Registry

from mmdet.models.builder import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
)

MODELS = Registry("models", parent=MMCV_MODELS)
SAMPLER = Registry("sampler")

VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS
FUSERS = Registry("fusers")


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """A function warpper for building 3D detector using cfg."""

    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
