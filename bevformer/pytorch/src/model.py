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
from .detr_head import DETRHead
from .neck import FPN
from .nms_freecoder import NMSFreeCoder
from PIL import Image
from .transformer import PerceptionTransformer

pts_bbox_head = {
    "type": "BEVFormerHead",
    "bev_h": 50,
    "bev_w": 50,
    "num_query": 900,
    "num_classes": 10,
    "in_channels": 256,
    "sync_cls_avg_factor": True,
    "with_box_refine": True,
    "as_two_stage": False,
    "transformer": {
        "type": "PerceptionTransformer",
        "rotate_prev_bev": True,
        "use_shift": True,
        "use_can_bus": True,
        "embed_dims": 256,
        "encoder": {
            "type": "BEVFormerEncoder",
            "num_layers": 3,
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "num_points_in_pillar": 4,
            "return_intermediate": False,
            "transformerlayers": {
                "type": "BEVFormerLayer",
                "attn_cfgs": [
                    {
                        "type": "TemporalSelfAttention",
                        "embed_dims": 256,
                        "num_levels": 1,
                    },
                    {
                        "type": "SpatialCrossAttention",
                        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        "deformable_attention": {
                            "type": "MSDeformableAttention3D",
                            "embed_dims": 256,
                            "num_points": 8,
                            "num_levels": 1,
                        },
                        "embed_dims": 256,
                    },
                ],
                "feedforward_channels": 512,
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
                        "embed_dims": 256,
                        "num_heads": 8,
                        "dropout": 0.1,
                    },
                    {
                        "type": "CustomMSDeformableAttention",
                        "embed_dims": 256,
                        "num_levels": 1,
                    },
                ],
                "feedforward_channels": 512,
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
        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "max_num": 300,
        "voxel_size": [0.2, 0.2, 8],
        "num_classes": 10,
    },
    "positional_encoding": {
        "type": "LearnedPositionalEncoding",
        "num_feats": 128,
        "row_num_embed": 50,
        "col_num_embed": 50,
    },
    "loss_cls": {
        "type": "FocalLoss",
        "use_sigmoid": True,
        "gamma": 2.0,
        "alpha": 0.25,
        "loss_weight": 2.0,
    },
    "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
    "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
    "train_cfg": None,
    "test_cfg": None,
}
img_backbone = {
    "type": "ResNet",
    "depth": 50,
    "num_stages": 4,
    "out_indices": (3,),
    "frozen_stages": 1,
    "norm_cfg": {"type": "BN", "requires_grad": False},
    "norm_eval": True,
    "style": "pytorch",
}
img_neck = {
    "type": "FPN",
    "in_channels": [2048],
    "out_channels": 256,
    "start_level": 0,
    "add_extra_convs": "on_output",
    "num_outs": 1,
    "relu_before_extra_convs": True,
}

data_test = {
    "type": "CustomNuScenesDataset",
    "data_root": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes",
    "ann_file": "/proj_sw/user_dev/mramanathan/bgdlab08_sep3_forge_new/tt-forge-fe/forge/test/models/pytorch/vision/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl",
    "pipeline": [
        {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
        {
            "type": "NormalizeMultiviewImage",
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "to_rgb": True,
        },
        {
            "type": "MultiScaleFlipAug3D",
            "img_scale": (1600, 900),
            "pts_scale_ratio": 1,
            "flip": False,
            "transforms": [
                {"type": "RandomScaleImageMultiViewImage", "scales": [0.5]},
                {"type": "PadMultiViewImage", "size_divisor": 32},
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
                {"type": "CustomCollect3D", "keys": ["img"]},
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
    "bev_size": (50, 50),
}


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
                img = img.squeeze(0)
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

    @property
    def volume(self):

        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):

        return self.tensor[:, 3:6]

    @property
    def yaw(self):

        return self.tensor[:, 6]

    @property
    def height(self):

        return self.tensor[:, 5]

    @property
    def top_height(self):

        return self.bottom_height + self.height

    @property
    def bottom_height(self):

        return self.tensor[:, 2]

    @property
    def center(self):

        return self.bottom_center

    @property
    def bottom_center(self):

        return self.tensor[:, :3]

    @property
    def gravity_center(self):

        pass

    @property
    def corners(self):

        pass

    @abstractmethod
    def rotate(self, angle, points=None):

        pass

    @abstractmethod
    def flip(self, bev_direction="horizontal"):

        pass

    def translate(self, trans_vector):

        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):

        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 2] > box_range[2])
            & (self.tensor[:, 0] < box_range[3])
            & (self.tensor[:, 1] < box_range[4])
            & (self.tensor[:, 2] < box_range[5])
        )
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):

        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):

        pass

    def scale(self, scale_factor):

        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):

        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0):

        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):

        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw,
            )
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self):

        return self.tensor.shape[0]

    def __repr__(self):

        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, boxes_list):
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].tensor.shape[1],
            with_yaw=boxes_list[0].with_yaw,
        )
        return cat_boxes

    def to(self, device):
        original_type = type(self)
        return original_type(
            self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw
        )

    def clone(self):

        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw
        )

    @property
    def device(self):

        return self.tensor.device

    def __iter__(self):

        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode="iou"):
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), (
            '"boxes1" and "boxes2" should'
            f"be in the same type, got {type(boxes1)} and {type(boxes2)}."
        )

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode="iou"):
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), (
            '"boxes1" and "boxes2" should'
            f"be in the same type, got {type(boxes1)} and {type(boxes2)}."
        )

        assert mode in ["iou", "iof"]

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        overlaps_h = cls.height_overlaps(boxes1, boxes2)

        # obtain BEV boxes in XYXYR format
        boxes1_bev = xywhr2xyxyr(boxes1.bev)
        boxes2_bev = xywhr2xyxyr(boxes2.bev)

        # bev overlap
        overlaps_bev = boxes1_bev.new_zeros(
            (boxes1_bev.shape[0], boxes2_bev.shape[0])
        ).cuda()  # (N, M)
        iou3d_cuda.boxes_overlap_bev_gpu(
            boxes1_bev.contiguous().cuda(), boxes2_bev.contiguous().cuda(), overlaps_bev
        )

        # 3d overlaps
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)

        if mode == "iou":
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d

    def new_box(self, data):
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, torch.Tensor)
            else data.to(self.device)
        )
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    @property
    def gravity_center(self):

        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        ).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self):

        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):

        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(
            conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4]
        )

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def rotate(self, angle, points=None):
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert (
            angle.shape == torch.Size([3, 3]) or angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[1, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # clockwise
                points.rotate(-angle)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction="horizontal", points=None):
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == "vertical":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 1] = -points[:, 1]
                elif bev_direction == "vertical":
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def in_range_bev(self, box_range):
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 0] < box_range[2])
            & (self.tensor[:, 1] < box_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        # from .box_3d_mode import Box3DMode
        return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def points_in_boxes(self, points):
        box_idx = points_in_boxes_gpu(
            points.unsqueeze(0), self.tensor.unsqueeze(0).to(points.device)
        ).squeeze(0)
        return box_idx


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
            # bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            bboxes = LiDARInstance3DBoxes(bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list


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
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        outs = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        return outs

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
