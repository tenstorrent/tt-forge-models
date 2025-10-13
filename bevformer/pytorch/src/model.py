# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from .backbone import ResNet
from .common_imports import BaseModule, Linear, auto_fp16, force_fp32
from .detr_head import DETRHead, NuscenesDD3D
from .neck import FPN
from .nms_freecoder import NMSFreeCoder
from PIL import Image
from .transformer import PerceptionTransformer
from collections import OrderedDict


def get_bevformer_v2_model(variant_str):
    """
    Get BEVFormerV2 model components based on variant string

    Args:
        variant_str: One of ['bevformerv2-r50-t1-base', 'bevformerv2-r50-t1', 'bevformerv2-r50-t2', 'bevformerv2-r50-t8']

    Returns:
        img_backbone, pts_bbox_head, img_neck, fcos3d_bbox_head, frames
    """
    # Base configuration
    base_config = {
        "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "class_names": [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
        ],
        "bev_h_": 200,
        "bev_w_": 200,
        "frames": (0,),
        "voxel_size": [102.4 / 200, 102.4 / 200, 8],
        "img_norm_cfg": dict(
            mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False
        ),
        "_dim_": 256,
        "_pos_dim_": 128,
        "_ffn_dim_": 512,
        "_num_levels_": 4,
        "_num_mono_levels_": 5,
        "ida_aug_conf": {
            "reisze": [
                640,
            ],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": False,
        },
        "ida_aug_conf_eval": {
            "reisze": [
                640,
            ],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": False,
        },
    }

    # Apply variant-specific modifications
    if variant_str == "bevformerv2-r50-t1-base":
        # Use base config as-is
        frames = base_config["frames"]
        group_detr = None
        pts_bbox_head_type = "BEVFormerHead"
        self_attn_type = "MultiheadAttention"

    elif variant_str == "bevformerv2-r50-t1":
        base_config["frames"] = (0,)
        base_config["group_detr"] = 11
        # Enhanced data augmentation for training
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    elif variant_str == "bevformerv2-r50-t2":
        base_config["frames"] = (
            -1,
            0,
        )
        base_config["group_detr"] = 11
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    elif variant_str == "bevformerv2-r50-t8":
        base_config["frames"] = (-7, -6, -5, -4, -3, -2, -1, 0)
        base_config["group_detr"] = 11
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    else:
        raise ValueError(
            f"Unknown variant: {variant_str}. Supported variants: 'bevformerv2-r50-t1-base', 'bevformerv2-r50-t1', 'bevformerv2-r50-t2', 'bevformerv2-r50-t8'"
        )

    # Common components
    img_backbone = {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (1, 2, 3),
        "frozen_stages": -1,
        "norm_cfg": {"type": "SyncBN"},
        "norm_eval": False,
        "style": "caffe",
    }

    img_neck = {
        "type": "FPN",
        "in_channels": [512, 1024, 2048],
        "out_channels": base_config["_dim_"],
        "start_level": 0,
        "add_extra_convs": "on_output",
        "num_outs": base_config["_num_mono_levels_"],
        "relu_before_extra_convs": True,
    }

    # Build pts_bbox_head based on variant
    if pts_bbox_head_type == "BEVFormerHead":
        pts_bbox_head = _get_base_pts_bbox_head(base_config)
    else:
        pts_bbox_head = _get_group_detr_pts_bbox_head(
            base_config, group_detr, self_attn_type
        )

    # FCOS3D bbox head (common across variants with minor adjustments)
    fcos3d_bbox_head = _get_fcos3d_bbox_head(base_config)

    return img_backbone, pts_bbox_head, img_neck, fcos3d_bbox_head, frames


def _get_base_pts_bbox_head(config):
    """Get base BEVFormerHead configuration"""
    return {
        "type": "BEVFormerHead",
        "bev_h": config["bev_h_"],
        "bev_w": config["bev_w_"],
        "num_query": 900,
        "num_classes": 10,
        "in_channels": config["_dim_"],
        "sync_cls_avg_factor": True,
        "with_box_refine": True,
        "as_two_stage": False,
        "transformer": {
            "type": "PerceptionTransformerV2",
            "embed_dims": config["_dim_"],
            "frames": config["frames"],
            "encoder": {
                "type": "BEVFormerEncoder",
                "num_layers": 6,
                "pc_range": config["point_cloud_range"],
                "num_points_in_pillar": 4,
                "return_intermediate": False,
                "transformerlayers": {
                    "type": "BEVFormerLayer",
                    "attn_cfgs": [
                        {
                            "type": "TemporalSelfAttention",
                            "embed_dims": config["_dim_"],
                            "num_levels": 1,
                        },
                        {
                            "type": "SpatialCrossAttention",
                            "pc_range": config["point_cloud_range"],
                            "deformable_attention": {
                                "type": "MSDeformableAttention3D",
                                "embed_dims": config["_dim_"],
                                "num_points": 8,
                                "num_levels": 4,
                            },
                            "embed_dims": config["_dim_"],
                        },
                    ],
                    "feedforward_channels": config["_ffn_dim_"],
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
            "decoder": {
                "type": "DetectionTransformerDecoder",
                "num_layers": 6,
                "return_intermediate": True,
                "transformerlayers": {
                    "type": "DetrTransformerDecoderLayer",
                    "attn_cfgs": [
                        {
                            "type": "MultiheadAttention",
                            "embed_dims": config["_dim_"],
                            "num_heads": 8,
                            "dropout": 0.1,
                        },
                        {
                            "type": "CustomMSDeformableAttention",
                            "embed_dims": config["_dim_"],
                            "num_levels": 1,
                        },
                    ],
                    "feedforward_channels": config["_ffn_dim_"],
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
        },
        "bbox_coder": {
            "type": "NMSFreeCoder",
            "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            "pc_range": config["point_cloud_range"],
            "max_num": 300,
            "voxel_size": config["voxel_size"],
            "num_classes": 10,
        },
        "positional_encoding": {
            "type": "LearnedPositionalEncoding",
            "num_feats": config["_pos_dim_"],
            "row_num_embed": config["bev_h_"],
            "col_num_embed": config["bev_w_"],
        },
        "loss_cls": {
            "type": "FocalLoss",
            "use_sigmoid": True,
            "gamma": 2.0,
            "alpha": 0.25,
            "loss_weight": 2.0,
        },
        "loss_bbox": {"type": "SmoothL1Loss", "loss_weight": 0.75, "beta": 1.0},
        "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
    }


def _get_group_detr_pts_bbox_head(config, group_detr, self_attn_type):
    """Get GroupDETR variant of BEVFormerHead"""
    pts_bbox_head = _get_base_pts_bbox_head(config)
    pts_bbox_head["type"] = "BEVFormerHead_GroupDETR"
    pts_bbox_head["group_detr"] = group_detr

    # Update decoder self-attention to GroupMultiheadAttention
    pts_bbox_head["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0] = {
        "type": self_attn_type,
        "group": group_detr,
        "embed_dims": config["_dim_"],
        "num_heads": 8,
        "dropout": 0.1,
    }

    # For t8 variant, add inter_channels
    if len(config["frames"]) > 2:
        pts_bbox_head["transformer"]["inter_channels"] = config["_dim_"] * 2

    return pts_bbox_head


def _get_fcos3d_bbox_head(config):
    """Get FCOS3D bbox head configuration"""
    return {
        "type": "NuscenesDD3D",
        "num_classes": 10,
        "in_channels": config["_dim_"],
        "strides": [8, 16, 32, 64, 128],
        "box3d_on": True,
        "feature_locations_offset": "none",
        "fcos2d_cfg": {
            "num_cls_convs": 4,
            "num_box_convs": 4,
            "norm": "SyncBN",
            "use_deformable": False,
            "use_scale": True,
            "box2d_scale_init_factor": 1.0,
        },
        "fcos2d_loss_cfg": {
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "loc_loss_type": "giou",
        },
        "fcos3d_cfg": {
            "num_convs": 4,
            "norm": "SyncBN",
            "use_scale": True,
            "depth_scale_init_factor": 0.3,
            "proj_ctr_scale_init_factor": 1.0,
            "use_per_level_predictors": False,
            "class_agnostic": False,
            "use_deformable": False,
            "mean_depth_per_level": [44.921, 20.252, 11.712, 7.166, 8.548],
            "std_depth_per_level": [24.331, 9.833, 6.223, 4.611, 8.275],
        },
        "fcos3d_loss_cfg": {
            "min_depth": 0.1,
            "max_depth": 80.0,
            "box3d_loss_weight": 2.0,
            "conf3d_loss_weight": 1.0,
            "conf_3d_temperature": 1.0,
            "smooth_l1_loss_beta": 0.05,
            "max_loss_per_group": 20,
            "predict_allocentric_rot": True,
            "scale_depth_by_focal_lengths": True,
            "scale_depth_by_focal_lengths_factor": 500.0,
            "class_agnostic": False,
            "predict_distance": False,
            "canon_box_sizes": [
                [2.3524184, 0.5062202, 1.0413622],
                [0.61416006, 1.7016163, 1.3054738],
                [2.9139307, 10.725025, 3.2832346],
                [1.9751819, 4.641267, 1.74352],
                [2.772134, 6.565072, 3.2474296],
                [0.7800532, 2.138673, 1.4437162],
                [0.6667362, 0.7181772, 1.7616143],
                [0.40246472, 0.4027083, 1.0084083],
                [3.0059454, 12.8197, 4.1213827],
                [2.4986045, 6.9310856, 2.8382742],
            ],
        },
        "target_assign_cfg": {
            "center_sample": True,
            "pos_radius": 1.5,
            "sizes_of_interest": [
                (-1, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 1e8),
            ],
        },
        "nusc_loss_weight": {"attr_loss_weight": 0.2, "speed_loss_weight": 0.2},
    }


class BEVFormerBaseConfig:
    """Base configuration for BEVFormer model components"""

    # Common parameters
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]

    class_names = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]

    # Default parameters (will be overridden by variants)
    _dim_ = 256
    _pos_dim_ = _dim_ // 2
    _ffn_dim_ = _dim_ * 2
    _num_levels_ = 4
    bev_h_ = 200
    bev_w_ = 200
    queue_length = 4

    # Base image normalization config
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
    )

    @classmethod
    def get_img_backbone(cls, variant="base"):
        """Get image backbone configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                type="ResNet",
                depth=50,
                num_stages=4,
                out_indices=(3,),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=False),
                norm_eval=True,
                style="pytorch",
            )
        elif variant == "BEVFormer-small":
            return dict(
                type="ResNet",
                depth=101,
                num_stages=4,
                out_indices=(3,),
                frozen_stages=1,
                norm_cfg=dict(type="BN2d", requires_grad=False),
                norm_eval=True,
                style="caffe",
                with_cp=True,
                dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            )
        else:  # base
            return dict(
                type="ResNet",
                depth=101,
                num_stages=4,
                out_indices=(1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN2d", requires_grad=False),
                norm_eval=True,
                style="caffe",
                dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            )

    @classmethod
    def get_img_neck(cls, variant="base"):
        """Get image neck configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                type="FPN",
                in_channels=[2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=1,
                relu_before_extra_convs=True,
            )
        elif variant == "BEVFormer-small":
            return dict(
                type="FPN",
                in_channels=[2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=1,
                relu_before_extra_convs=True,
            )
        else:  # base
            return dict(
                type="FPN",
                in_channels=[512, 1024, 2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=4,
                relu_before_extra_convs=True,
            )

    @classmethod
    def get_pts_bbox_head(cls, variant="base"):
        """Get BEVFormer head configuration"""
        # Get variant-specific parameters
        params = cls._get_variant_params(variant)

        # Base bbox head configuration
        bbox_head = dict(
            type="BEVFormerHead",
            bev_h=params["bev_h"],
            bev_w=params["bev_w"],
            num_query=900,
            num_classes=10,
            in_channels=cls._dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type="PerceptionTransformer",
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=cls._dim_,
                encoder=dict(
                    type="BEVFormerEncoder",
                    num_layers=params["encoder_layers"],
                    pc_range=cls.point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(
                                type="TemporalSelfAttention",
                                embed_dims=cls._dim_,
                                num_levels=1,
                            ),
                            dict(
                                type="SpatialCrossAttention",
                                pc_range=cls.point_cloud_range,
                                deformable_attention=dict(
                                    type="MSDeformableAttention3D",
                                    embed_dims=cls._dim_,
                                    num_points=8,
                                    num_levels=params["num_levels"],
                                ),
                                embed_dims=cls._dim_,
                            ),
                        ],
                        feedforward_channels=cls._ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
                decoder=dict(
                    type="DetectionTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=cls._dim_,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type="CustomMSDeformableAttention",
                                embed_dims=cls._dim_,
                                num_levels=1,
                            ),
                        ],
                        feedforward_channels=cls._ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=cls.point_cloud_range,
                max_num=300,
                voxel_size=cls.voxel_size,
                num_classes=10,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=cls._pos_dim_,
                row_num_embed=params["bev_h"],
                col_num_embed=params["bev_w"],
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        )

        return bbox_head

    @classmethod
    def _get_variant_params(cls, variant):
        """Get variant-specific parameters"""
        if variant == "BEVFormer-tiny":
            return {
                "bev_h": 50,
                "bev_w": 50,
                "encoder_layers": 3,
                "num_levels": 1,
                "queue_length": 3,
            }
        elif variant == "BEVFormer-small":
            return {
                "bev_h": 150,
                "bev_w": 150,
                "encoder_layers": 3,
                "num_levels": 1,
                "queue_length": 3,
            }
        else:  # base
            return {
                "bev_h": 200,
                "bev_w": 200,
                "encoder_layers": 6,
                "num_levels": 4,
                "queue_length": 4,
            }

    @classmethod
    def get_img_norm_cfg(cls, variant="base"):
        """Get image normalization configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
            )
        else:  # BEVFormer-small and BEVFormer-base
            return cls.img_norm_cfg


def get_bevformer_model(variant="BEVFormer-tiny"):
    """
    Get BEVFormer model components for the specified variant

    Args:
        variant (str): Model variant - 'BEVFormer-tiny', 'BEVFormer-small', or 'BEVFormer-base'

    Returns:
        tuple: (img_backbone, pts_bbox_head, img_neck) configurations
    """
    valid_variants = [
        "BEVFormer-tiny",
        "BEVFormer-small",
        "BEVFormer-base",
        "BEVFormerV2-r50-t1-base",
    ]
    if variant not in valid_variants:
        raise ValueError(f"Variant must be one of {valid_variants}, got {variant}")

    # Get configurations for the specified variant
    img_backbone = BEVFormerBaseConfig.get_img_backbone(variant)
    img_neck = BEVFormerBaseConfig.get_img_neck(variant)
    pts_bbox_head = BEVFormerBaseConfig.get_pts_bbox_head(variant)

    return img_backbone, pts_bbox_head, img_neck


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu()
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@classmethod
def load_checkpoint(cls, filename, map_location=None, logger=None):

    checkpoint_loader = cls._get_checkpoint_loader(filename)
    class_name = checkpoint_loader.__name__
    print(f"load checkpoint from {class_name[10:]} path: {filename}")
    return checkpoint_loader(filename, map_location)


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class BaseDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) "
                f"!= num of image meta ({len(img_metas)})"
            )

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if "proposals" in kwargs:
                kwargs["proposals"] = kwargs["proposals"][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, (
                "aug test does not support "
                "inference with batch size "
                f"{imgs[0].size(0)}"
            )
            # TODO: support test augmentation for predefined proposals
            assert "proposals" not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(
            f"{self.__class__.__name__} does " f"not support ONNX EXPORT"
        )


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(points), len(img_metas)
                )
            )

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    # @auto_fp16(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


class MVXTwoStageDetector(Base3DDetector):
    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            trans_type = pts_bbox_head.get("type", None)
            if trans_type == "BEVFormerHead_GroupDETR":
                cfg = pts_bbox_head.copy()
                self.pts_bbox_head = BEVFormerHead_GroupDETR(**cfg)
            else:
                cfg = pts_bbox_head.copy()
                self.pts_bbox_head = BEVFormerHead(**cfg)
            # if isinstance(pts_bbox_head, dict):
            #     cfg = pts_bbox_head.copy()
            #     head_type = cfg.pop("type", None)
            #     # Both BEVFormerHead and BEVFormerHead_GroupDETR map to BEVFormerHead implementation
            #     self.pts_bbox_head = BEVFormerHead(**cfg)
            # elif isinstance(pts_bbox_head, BEVFormerHead_GroupDETR):
            #     # Treat BEVFormerHead_GroupDETR the same as BEVFormerHead
            #     self.pts_bbox_head = pts_bbox_head
            # elif isinstance(pts_bbox_head, BEVFormerHead):
            #     self.pts_bbox_head = pts_bbox_head
            # else:
            #     raise TypeError(
            #         "pts_bbox_head must be a dict config or a BEVFormerHead instance"
            #     )
        if img_backbone:
            if isinstance(img_backbone, dict):
                cfg = img_backbone.copy()
                cfg.pop("type", None)
                self.img_backbone = ResNet(**cfg)
            elif isinstance(img_backbone, ResNet):
                self.img_backbone = img_backbone
            else:
                raise TypeError(
                    "img_backbone must be a dict config or a ResNet instance"
                )
        if img_neck is not None:
            if isinstance(img_neck, dict):
                cfg = img_neck.copy()
                cfg.pop("type", None)
                self.img_neck = FPN(**cfg)
            elif isinstance(img_neck, FPN):
                self.img_neck = img_neck
            else:
                raise TypeError("img_neck must be a dict config or an FPN instance")

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get("img", None)
            pts_pretrained = pretrained.get("pts", None)
        else:
            raise ValueError(f"pretrained should be a dict, got {type(pretrained)}")

    @property
    def with_pts_bbox(self):
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        return hasattr(self, "img_bbox_head") and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_neck(self):
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        return hasattr(self, "img_rpn_head") and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    def extract_img_feat(self, img, img_metas):
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict["img_bbox"] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]


class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    # @auto_fp16()
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).cuda()
            )
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask.cpu()

        return x.view(n, c, h, w)


class BEVFormerHead(DETRHead):
    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        **kwargs,
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        if isinstance(bbox_coder, dict):
            cfg = bbox_coder.copy()
            cfg.pop("type", None)
            self.bbox_coder = NMSFreeCoder(**cfg)
        elif isinstance(bbox_coder, NMSFreeCoder):
            self.bbox_coder = bbox_coder
        else:
            raise TypeError(
                "bbox_coder must be a dict config or an NMSFreeCoder instance"
            )
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    # @auto_fp16(apply_to=("mlvl_feats"))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        # return mlvl_feats
        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if (
            only_bev
        ):  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches
                if self.with_box_refine
                else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        # return outputs
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs

    # @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list


class BEVFormerHead_GroupDETR(BEVFormerHead):
    def __init__(self, *args, group_detr=1, **kwargs):
        self.group_detr = group_detr
        assert "num_query" in kwargs
        kwargs["num_query"] = group_detr * kwargs["num_query"]
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[
                : self.num_query // self.group_detr
            ]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches
                if self.with_box_refine
                else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs

    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
        img_metas=None,
    ):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        assert enc_cls_scores is None and enc_bbox_preds is None

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        loss_dict = dict()
        loss_dict["loss_cls"] = 0
        loss_dict["loss_bbox"] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = 0
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = 0
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index + 1) * num_query_per_group
            group_cls_scores = all_cls_scores[
                :, :, group_query_start:group_query_end, :
            ]
            group_bbox_preds = all_bbox_preds[
                :, :, group_query_start:group_query_end, :
            ]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                group_cls_scores,
                group_bbox_preds,
                all_gt_bboxes_list,
                all_gt_labels_list,
                all_gt_bboxes_ignore_list,
            )
            loss_dict["loss_cls"] += losses_cls[-1] / self.group_detr
            loss_dict["loss_bbox"] += losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f"d{num_dec_layer}.loss_cls"] += loss_cls_i / self.group_detr
                loss_dict[f"d{num_dec_layer}.loss_bbox"] += (
                    loss_bbox_i / self.group_detr
                )
                num_dec_layer += 1
        return loss_dict


class BEVFormerV2(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        fcos3d_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        num_levels=None,
        num_mono_levels=None,
        mono_loss_weight=1.0,
        frames=(0,),
    ):

        super(BEVFormerV2, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        assert not self.fp16_enabled  # not support fp16 yet
        # temporal
        self.video_test_mode = video_test_mode
        assert not self.video_test_mode  # not support video_test_mode yet

        # fcos3d head
        if isinstance(fcos3d_bbox_head, dict):
            cfg = fcos3d_bbox_head.copy()
            cfg.pop("type", None)
            self.fcos3d_bbox_head = NuscenesDD3D(**cfg)
        # self.fcos3d_bbox_head = build_head(fcos3d_bbox_head) if fcos3d_bbox_head else None
        # loss weight
        self.mono_loss_weight = mono_loss_weight

        # levels of features
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img)
        if (
            "aug_param" in img_metas[0]
            and img_metas[0]["aug_param"]["CropResizeFlipImage_param"][-1] is True
        ):
            # flip feature
            img_feats = [
                torch.flip(
                    x,
                    dims=[
                        -1,
                    ],
                )
                for x in img_feats
            ]
        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None,
    ):
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_mono_train(self, img_feats, mono_input_dict):
        """
        img_feats (list[Tensor]): 5-D tensor for each level, (B, N, C, H, W)
        gt_bboxes (list[list[Tensor]]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels (list[list[Tensor]]): class indices corresponding to each box
        gt_bboxes_3d (list[list[[Tensor]]): 3D boxes ground truth with shape of
                (num_gts, code_size).
        gt_labels_3d (list[list[Tensor]]): same as gt_labels
        centers2d (list[list[Tensor]]): 2D centers on the image with shape of
                (num_gts, 2).
        depths (list[list[Tensor]]): Depth ground truth with shape of
                (num_gts, ).
        attr_labels (list[list[Tensor]]): Attributes indices of each box.
        img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
        ann_idx (list[list[idx]]): indicate which image has mono annotation.
        """
        bsz = img_feats[0].shape[0]
        num_lvls = len(img_feats)

        img_feats_select = [[] for lvl in range(num_lvls)]
        for lvl, img_feat in enumerate(img_feats):
            for i in range(bsz):
                img_feats_select[lvl].append(
                    img_feat[i, mono_input_dict["mono_ann_idx"][i]]
                )
            img_feats_select[lvl] = torch.cat(img_feats_select[lvl], dim=0)
        bsz_new = img_feats_select[0].shape[0]
        assert bsz == len(mono_input_dict["mono_input_dict"])
        input_dict = []
        for i in range(bsz):
            input_dict.extend(mono_input_dict["mono_input_dict"][i])
        assert bsz_new == len(input_dict)
        losses = self.fcos3d_bbox_head.forward_train(img_feats_select, input_dict)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, img_dict, img_metas_dict):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        prev_bev = OrderedDict({i: None for i in self.frames})
        with torch.no_grad():
            for t in img_dict.keys():
                img = img_dict[t]
                img_metas = [
                    img_metas_dict[t],
                ]
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
                if self.num_levels:
                    img_feats = img_feats[: self.num_levels]
                bev = self.pts_bbox_head(img_feats, img_metas, None, only_bev=True)
                prev_bev[t] = bev.detach()
        if is_training:
            self.train()
        return list(prev_bev.values())

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        gt_bboxes_ignore=None,
        **mono_input_dict,
    ):
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]

        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(0)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [
            img_metas[0],
        ]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats if self.num_levels is None else img_feats[: self.num_levels],
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
        )
        losses.update(losses_pts)

        if self.fcos3d_bbox_head:
            losses_mono = self.forward_mono_train(
                img_feats=img_feats
                if self.num_mono_levels is None
                else img_feats[: self.num_mono_levels],
                mono_input_dict=mono_input_dict,
            )
            for k, v in losses_mono.items():
                losses[f"{k}_mono"] = v * self.mono_loss_weight

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=None, **kwargs
        )
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [
            img_metas[0],
        ]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.num_levels:
            img_feats = img_feats[: self.num_levels]

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list


class BEVFormer(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):

        super(BEVFormer, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
                # img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    # @auto_fp16(apply_to=("img"))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward(self, return_loss=True, **kwargs):

        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # return img_feats
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list
