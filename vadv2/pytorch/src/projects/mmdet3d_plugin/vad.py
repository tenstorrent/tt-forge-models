# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
from third_party.tt_forge_models.vadv2.pytorch.src.build_model import DETECTORS
from abc import ABCMeta, abstractmethod
from scipy.optimize import linear_sum_assignment
from .BaseModule import BaseModule
from .vad_utils import multi_apply
from ...build_model import build_head, build_backbone, build_neck
import warnings
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from skimage.draw import polygon

ego_width, ego_length = 1.85, 4.084


class GridMask(nn.Module):
    def __init__(
        self,
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
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

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def _rotate_tensor(self, tensor, angle):
        """
        Rotate a 2D tensor by the given angle using PyTorch operations.
        Args:
            tensor: Input tensor of shape (H, W)
            angle: Rotation angle in degrees
        Returns:
            Rotated tensor of same shape
        """
        # Ensure tensor is 3D (C, H, W) for interpolation
        tensor = tensor.unsqueeze(0)  # Shape: (1, H, W)
        # Convert angle to radians
        angle_rad = torch.tensor(angle * np.pi / 180.0, device=tensor.device)
        # Create rotation matrix
        cos_val = torch.cos(angle_rad)
        sin_val = torch.sin(angle_rad)
        rotation_matrix = torch.tensor(
            [[cos_val, -sin_val, 0], [sin_val, cos_val, 0]], device=tensor.device
        )
        # Create grid for affine transformation
        grid = F.affine_grid(
            rotation_matrix.unsqueeze(0),
            size=(1, 1, tensor.shape[1], tensor.shape[2]),
            align_corners=False,
        )
        # Apply rotation
        rotated_tensor = F.grid_sample(
            tensor.unsqueeze(0), grid, align_corners=False, mode="bilinear"
        )
        return rotated_tensor.squeeze(0).squeeze(0)  # Shape: (H, W)

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)  # Flatten batch and channel for processing
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, min(h, w))
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # Create mask
        mask = torch.ones((hh, ww), dtype=x.dtype, device=x.device)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0

        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        # Rotate mask if needed
        if self.rotate > 0:
            r = np.random.randint(self.rotate)
            mask = self._rotate_tensor(mask, r)

        # Crop mask to original image size
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        # Apply mode (0: mask zeros, 1: mask ones)
        if self.mode == 1:
            mask = 1 - mask

        # Expand mask to match input tensor shape
        mask = mask.expand_as(x)

        # Apply mask to input
        if self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .to(x.dtype)
                .to(x.device)
            )
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


class PlanningMetric:
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        (
            bev_resolution,
            bev_start_position,
            bev_dimension,
        ) = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        self.category_index = {
            "human": [2, 3, 4, 5, 6, 7, 8],
            "vehicle": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor(
            [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
        )

        return dx, bx, nx

    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor(
            [row[2] for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_start_position = torch.tensor(
            [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_dimension = torch.tensor(
            [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
            dtype=torch.long,
        )

        return bev_resolution, bev_start_position, bev_dimension

    def get_label(self, gt_agent_boxes, gt_agent_feats):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(
            gt_agent_boxes, gt_agent_feats
        )
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian

    def get_birds_eye_view_label(self, gt_agent_boxes, gt_agent_feats):
        """
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9)
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        """
        T = 6
        segmentation = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.cpu().numpy()  # (N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., : T * 2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T * 2 : T * 3].reshape(-1, 6)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., T * 3 + 10 : T * 4 + 10].reshape(
            -1, 6, 1
        )
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:, 6:7] = -1 * (
            gt_agent_boxes[:, 6:7] + np.pi / 2
        )  # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]

        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[0, i][27])
                    agent_length, agent_width = (
                        gt_agent_boxes[i][4],
                        gt_agent_boxes[i][3],
                    )
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a, y_a, yaw_a, agent_length, agent_width]
                    if category_index in self.category_index["vehicle"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if category_index in self.category_index["human"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)

        return segmentation, pedestrian

    def _get_poly_region_in_image(self, param):
        lidar2cv_rot = np.array([[1, 0], [0, -1]])
        x_a, y_a, yaw_a, agent_length, agent_width = param
        trans_a = np.array([[x_a, y_a]]).T
        rot_mat_a = np.array(
            [[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]]
        )
        agent_corner = np.array(
            [
                [
                    agent_length / 2,
                    -agent_length / 2,
                    -agent_length / 2,
                    agent_length / 2,
                ],
                [agent_width / 2, agent_width / 2, -agent_width / 2, -agent_width / 2],
            ]
        )  # (2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a  # (2,4)
        # convert to cv frame
        agent_corner_cv2 = (
            np.matmul(lidar2cv_rot, agent_corner_lidar)
            - self.bev_start_position[:2, None]
            + self.bev_resolution[:2, None] / 2.0
        ).T / self.bev_resolution[
            :2
        ]  # (4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2

    def evaluate_single_coll(self, traj, segmentation, input_gt):
        """
        traj: torch.Tensor (n_future, 2)
            自车lidar系为轨迹参考系
                ^ y
                |
                |
                0------->
                        x
        segmentation: torch.Tensor (n_future, 200, 200)
        """
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # 轨迹坐标系转换为:
        #  ^ x
        #  |
        #  |
        #  0-------> y
        trajs_ = copy.deepcopy(trajs)
        trajs_[:, :, [0, 1]] = trajs_[:, :, [1, 0]]  # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc  # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:, :, 0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        trajs: torch.Tensor (B, n_future, 2)
            自车lidar系为轨迹参考系
            ^ y
            |
            |
            0------->
                    x
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        """
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)
        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(
                gt_trajs[i], segmentation[i], input_gt=True
            )

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            # lidar系下的轨迹转换到图片坐标系下
            xi = ((-self.bx[0] / 2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1] / 2 + xx) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(
                trajs[i], segmentation[i], input_gt=False
            ).to(ti.device)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        """
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        """
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )

        return ade


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) "
                f"!= num of image meta ({len(img_metas)})"
            )

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

        if num_augs == 1:
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

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """

        return self.forward_test(img, img_metas, **kwargs)


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
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

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

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
            self.pts_bbox_head = build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)

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

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                self.img_backbone.init_cfg = dict(
                    type="Pretrained", checkpoint=img_pretrained
                )

        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                self.pts_backbone.init_cfg = dict(
                    type="Pretrained", checkpoint=pts_pretrained
                )

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()

        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu()
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model."""

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
        fut_ts=6,
        fut_mode=6,
    ):

        super(VAD, self).__init__(
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
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head["valid_fut_ts"]

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.planning_metric = None

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

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
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward(self, return_loss=True, **kwargs):
        device = torch.device("cpu")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device)
            elif isinstance(v, list):
                kwargs[k] = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in v
                ]

        torch.save(kwargs, "input_cpu.pth")
        return self.forward_test(**kwargs)

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0][0]["can_bus"][-1] = 0
            img_metas[0][0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0][0],
            img=img[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs,
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
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

        outs = self.pts_bbox_head(
            x,
            img_metas,
            prev_bev=prev_bev,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
        )
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for (
            i,
            (
                bboxes,
                scores,
                labels,
                trajs,
                map_bboxes,
                map_scores,
                map_labels,
                map_pts,
            ),
        ) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.map_pred2result(
                map_bboxes, map_scores, map_labels, map_pts
            )
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_result["ego_fut_cmd"] = ego_fut_cmd
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "only support batch_size=1 now"
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0][0]
            gt_label = gt_labels_3d[0][0][0]
            gt_attr_label = gt_attr_labels[0][0][0]
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            mask = bbox_result["scores_3d"] > score_threshold
            bbox_result["boxes_3d"] = bbox_result["boxes_3d"][mask]
            bbox_result["scores_3d"] = bbox_result["scores_3d"][mask]
            bbox_result["labels_3d"] = bbox_result["labels_3d"][mask]
            bbox_result["trajs_3d"] = bbox_result["trajs_3d"][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label
            )

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox,
                gt_label,
                gt_attr_label,
                bbox_result,
                matched_bbox_result,
                mapped_class_names,
            )

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, "only support batch_size=1 for testing"
            ego_fut_preds = bbox_result["ego_fut_preds"]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs=ego_fut_pred[None],
                gt_ego_fut_trajs=ego_fut_trajs[None],
                gt_agent_boxes=gt_bbox,
                gt_agent_feats=gt_attr_label.unsqueeze(0),
                fut_valid_flag=fut_valid_flag,
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs["bev_embed"], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to("cpu"),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to("cpu"),
        )

        if attrs is not None:
            result_dict["map_attrs_3d"] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self, bbox_result, gt_bbox, gt_label, match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """
        dynamic_list = [0, 1, 3, 4, 6, 7, 8]
        matched_bbox_result = (
            torch.ones((len(gt_bbox)), dtype=torch.long) * -1
        )  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result["boxes_3d"].center[:, :2]
        dist = torch.linalg.norm(
            pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1
        )
        pred_not_dyn = [label not in dynamic_list for label in bbox_result["labels_3d"]]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ["car", "pedestrian"]
        motion_metric_names = [
            "gt",
            "cnt_ade",
            "cnt_fde",
            "hit",
            "fp",
            "ADE",
            "FDE",
            "MR",
        ]

        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met + "_" + cls] = 0.0

        veh_list = [0, 1, 3, 4]
        ignore_list = [
            "construction_vehicle",
            "barrier",
            "traffic_cone",
            "motorcycle",
            "bicycle",
        ]

        for i in range(pred_bbox["labels_3d"].shape[0]):
            pred_bbox["labels_3d"][i] = (
                0
                if pred_bbox["labels_3d"][i] in veh_list
                else pred_bbox["labels_3d"][i]
            )
            box_name = mapped_class_names[pred_bbox["labels_3d"][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict["fp_" + box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts * 2 : self.fut_ts * 3]
            num_valid_ts = sum(gt_fut_masks == 1)
            if num_valid_ts == self.fut_ts:
                metric_dict["gt_" + box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict["cnt_ade_" + box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][: self.fut_ts * 2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox["trajs_3d"][m_pred_idx].reshape(
                    self.fut_mode, self.fut_ts, 2
                )
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = (
                    pred_fut_trajs
                    + pred_bbox["boxes_3d"][int(m_pred_idx)].center[0, :2]
                )

                dist = torch.linalg.norm(
                    gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1
                )
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict["ADE_" + box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict["cnt_fde_" + box_name] += 1
                    metric_dict["FDE_" + box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict["hit_" + box_name] += 1
                    else:
                        metric_dict["MR_" + box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag,
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            "plan_L2_1s": 0,
            "plan_L2_2s": 0,
            "plan_L2_3s": 0,
            "plan_obj_col_1s": 0,
            "plan_obj_col_2s": 0,
            "plan_obj_col_3s": 0,
            "plan_obj_box_col_1s": 0,
            "plan_obj_box_col_2s": 0,
            "plan_obj_box_col_3s": 0,
        }
        metric_dict["fut_valid_flag"] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, "only support bs=1"
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats
        )
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time]
                    .detach()
                    .to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time],
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy,
                )
                metric_dict["plan_L2_{}s".format(i + 1)] = traj_L2
                metric_dict["plan_obj_col_{}s".format(i + 1)] = obj_coll.mean().item()
                metric_dict[
                    "plan_obj_box_col_{}s".format(i + 1)
                ] = obj_box_coll.mean().item()
            else:
                metric_dict["plan_L2_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_col_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_box_col_{}s".format(i + 1)] = 0.0

        return metric_dict
