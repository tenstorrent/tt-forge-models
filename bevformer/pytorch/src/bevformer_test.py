# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from .backbone import ResNet, BaseModule, Linear
from .detr_head import DETRHead, NuscenesDD3D
from .neck import FPN
from .nms_freecoder import NMSFreeCoder
from PIL import Image
from collections import OrderedDict
from .transformer import PerceptionTransformer

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
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
]
dataset_type = "CustomNuScenesDatasetV2"
data_root = "data/nuscenes/"
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False)
bev_h_ = 200
bev_w_ = 200
frames = (0,)
voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8]
ida_aug_conf = {
    "reisze": [
        640,
    ],
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
ida_aug_conf_eval = {
    "reisze": [
        640,
    ],
    "crop": (0, 260, 1600, 900),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_mono_levels_ = 5

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
    "out_channels": _dim_,
    "start_level": 0,
    "add_extra_convs": "on_output",
    "num_outs": _num_mono_levels_,
    "relu_before_extra_convs": True,
}
pts_bbox_head = {
    "type": "BEVFormerHead",
    "bev_h": bev_h_,
    "bev_w": bev_w_,
    "num_query": 900,
    "num_classes": 10,
    "in_channels": _dim_,
    "sync_cls_avg_factor": True,
    "with_box_refine": True,
    "as_two_stage": False,
    "transformer": {
        "type": "PerceptionTransformerV2",
        "embed_dims": _dim_,
        "frames": frames,
        "encoder": {
            "type": "BEVFormerEncoder",
            "num_layers": 6,
            "pc_range": point_cloud_range,
            "num_points_in_pillar": 4,
            "return_intermediate": False,
            "transformerlayers": {
                "type": "BEVFormerLayer",
                "attn_cfgs": [
                    {
                        "type": "TemporalSelfAttention",
                        "embed_dims": _dim_,
                        "num_levels": 1,
                    },
                    {
                        "type": "SpatialCrossAttention",
                        "pc_range": point_cloud_range,
                        "deformable_attention": {
                            "type": "MSDeformableAttention3D",
                            "embed_dims": _dim_,
                            "num_points": 8,
                            "num_levels": 4,
                        },
                        "embed_dims": _dim_,
                    },
                ],
                "feedforward_channels": _ffn_dim_,
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
                        "embed_dims": _dim_,
                        "num_heads": 8,
                        "dropout": 0.1,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": _dim_,
                        "num_levels": 1,
                    },
                ],
                "feedforward_channels": _ffn_dim_,
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
        "pc_range": point_cloud_range,
        "max_num": 300,
        "voxel_size": voxel_size,
        "num_classes": 10,
    },
    "positional_encoding": {
        "type": "LearnedPositionalEncoding",
        "num_feats": _pos_dim_,
        "row_num_embed": bev_h_,
        "col_num_embed": bev_w_,
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


fcos3d_bbox_head = {
    "type": "NuscenesDD3D",
    "num_classes": 10,
    "in_channels": _dim_,
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
        "sizes_of_interest": (
            (-1, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 100000000.0),
        ),
    },
    "nusc_loss_weight": {
        "attr_loss_weight": 0.2,
        "speed_loss_weight": 0.2,
    },
}

data_test = {
    "type": "CustomNuScenesDatasetV2",
    "data_root": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes",
    "ann_file": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl",
    "pipeline": [
        {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
        {
            "type": "CropResizeFlipImage",
            "data_aug_conf": ida_aug_conf_eval,
            "training": False,
            "debug": False,
        },
        {
            "type": "NormalizeMultiviewImage",
            "mean": [
                103.53,
                116.28,
                123.675,
            ],  # replace with values in img_norm_cfg if different
            "std": [1, 1, 1],
            "to_rgb": False,  # add if applicable from img_norm_cfg
        },
        {"type": "PadMultiViewImage", "size_divisor": 32},
        {
            "type": "MultiScaleFlipAug3D",
            "img_scale": (1600, 640),
            "pts_scale_ratio": 1,
            "flip": False,
            "transforms": [
                {
                    "type": "DefaultFormatBundle3D",
                    "class_names": [
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
                    ],
                    "with_label": False,
                },
                {
                    "type": "CustomCollect3D",
                    "keys": [
                        "img",
                        "ego2global_translation",
                        "ego2global_rotation",
                        "lidar2ego_translation",
                        "lidar2ego_rotation",
                        "timestamp",
                    ],
                },
            ],
        },
    ],
    "classes": [
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
    ],
    "modality": {
        "use_lidar": False,
        "use_camera": True,
        "use_radar": False,
        "use_map": False,
        "use_external": True,
    },
    "test_mode": True,
    "box_type_3d": "LiDAR",
}
# data_test = {
#     "type": "CustomNuScenesDataset",
#     "data_root": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes",
#     "ann_file": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl",
#     "pipeline": [
#         {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
#         {
#             "type": "NormalizeMultiviewImage",
#             "mean": [123.675, 116.28, 103.53],
#             "std": [58.395, 57.12, 57.375],
#             "to_rgb": True,
#         },
#         {
#             "type": "MultiScaleFlipAug3D",
#             "img_scale": (1600, 900),
#             "pts_scale_ratio": 1,
#             "flip": False,
#             "transforms": [
#                 {"type": "RandomScaleImageMultiViewImage", "scales": [0.5]},
#                 {"type": "PadMultiViewImage", "size_divisor": 32},
#                 {
#                     "type": "DefaultFormatBundle3D",
#                     "class_names": [
#                         "car",
#                         "truck",
#                         "construction_vehicle",
#                         "bus",
#                         "trailer",
#                         "barrier",
#                         "motorcycle",
#                         "bicycle",
#                         "pedestrian",
#                         "traffic_cone",
#                     ],
#                     "with_label": False,
#                 },
#                 {"type": "CustomCollect3D", "keys": ["img"]},
#             ],
#         },
#     ],
#     "classes": [
#         "car",
#         "truck",
#         "construction_vehicle",
#         "bus",
#         "trailer",
#         "barrier",
#         "motorcycle",
#         "bicycle",
#         "pedestrian",
#         "traffic_cone",
#     ],
#     "modality": {
#         "use_lidar": False,
#         "use_camera": True,
#         "use_radar": False,
#         "use_map": False,
#         "use_external": True,
#     },
#     "test_mode": True,
#     "box_type_3d": "LiDAR",
#     "bev_size": (50, 50),
# }


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

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        return self.forward_test(img, img_metas, **kwargs)


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward(self, return_loss=True, **kwargs):
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
            if isinstance(pts_bbox_head, dict):
                cfg = pts_bbox_head.copy()
                cfg.pop("type", None)
                self.pts_bbox_head = BEVFormerHead(**cfg)
            elif isinstance(pts_bbox_head, BEVFormerHead):
                self.pts_bbox_head = pts_bbox_head
            else:
                raise TypeError(
                    "pts_bbox_head must be a dict config or a BEVFormerHead instance"
                )
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

    @property
    def with_img_neck(self):
        return hasattr(self, "img_neck") and self.img_neck is not None


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
        x = x * mask.cpu()

        return x.view(n, c, h, w)


class BaseInstance3DBoxes(object):
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __repr__(self):
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    def to(self, device):
        original_type = type(self)
        return original_type(
            self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw
        )


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    @property
    def bev(self):
        return self.tensor[:, [0, 1, 3, 4, 6]]


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

        self.code_size = kwargs.get("code_size", 10)
        self.code_weights = [1.0] * 8 + [0.2, 0.2]

        if isinstance(bbox_coder, dict):
            cfg = bbox_coder.copy()
            cfg.pop("type", None)
            self.bbox_coder = NMSFreeCoder(**cfg)

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
            cls_branch += [
                Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            ]
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch += [Linear(self.embed_dims, self.embed_dims), nn.ReLU()]
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        assert isinstance(mlvl_feats, list), "mlvl_feats must be a list of tensors"
        assert isinstance(img_metas, list), "img_metas must be a list"

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert (
                reference.shape[-1] == 3
            ), f"Unexpected reference shape {reference.shape}"

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

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }
        return outs

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        ret_list = []
        for i, preds in enumerate(preds_dicts):
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = LiDARInstance3DBoxes(bboxes, code_size)
            scores, labels = preds["scores"], preds["labels"]
            ret_list.append([bboxes, scores, labels])

        return ret_list


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
        breakpoint()
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
                img = img.squeeze(0)
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

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward(self, return_loss=True, **kwargs):
        print("inputs = ", kwargs)
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
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list
