# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ...build_model import BACKBONES, LOSSES, NECKS
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import functools
import math
import numpy as np
import torch.distributed as dist
from functools import partial
import copy
import torch.nn as nn


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@BACKBONES.register_module()
class ResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_stages,
        out_indices,
        frozen_stages,
        norm_cfg,
        norm_eval,
        style,
        pretrained=True,
    ):
        super().__init__()
        # Load pretrained ResNet50
        base_model = models.resnet50(pretrained=True)

        # Copy layers except avgpool & fc
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        outs.append(x)
        return tuple(outs)  # Shape: (B, 2048, H/32, W/32)


class L1Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None
):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = (
        F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        * focal_weight
    )
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):
    def __init__(
        self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0
    ):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:

            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
            )

        else:
            raise NotImplementedError
        return loss_cls


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(
            bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
        )  # [B, rows, cols, 2]
        rb = torch.min(
            bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
        )  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(
                bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
            )
            enclosed_rb = torch.max(
                bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
            )

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


class GIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


@weighted_loss
def pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


class PtsL1Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


def custom_weighted_dir_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction="none")
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


class PtsDirCosLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_dir


@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    pred = pred.cumsum(dim=-2)
    ego_traj_starts = pred[:, :-1, :]
    ego_traj_ends = pred
    B, T, _ = ego_traj_ends.size()
    padding_zeros = torch.zeros(
        (B, 1, 2), dtype=pred.dtype, device=pred.device
    )  # initial position
    ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
    _, V, P, _ = target.size()
    ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 2]
    maps_expanded = target.unsqueeze(1)  # [1, 1, M, P, 2]
    dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [B, T, M, P]
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    bd_target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)
    min_bd_insts = bd_target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, T, P, 2]
    bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
    bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
    ego_traj_starts = ego_traj_starts.unsqueeze(2).repeat(1, 1, P - 1, 1).flatten(0, 2)
    ego_traj_ends = ego_traj_ends.unsqueeze(2).repeat(1, 1, P - 1, 1).flatten(0, 2)

    intersect_mask = segments_intersect(
        ego_traj_starts, ego_traj_ends, bd_inst_starts, bd_inst_ends
    )
    intersect_mask = intersect_mask.reshape(B, T, P - 1)
    intersect_mask = intersect_mask.any(dim=-1)
    intersect_idx = (intersect_mask == True).nonzero()

    target = target.view(target.shape[0], -1, target.shape[-1])
    # [B, fut_ts, num_vec*num_pts]
    dist = torch.linalg.norm(pred[:, :, None, :] - target[:, None, :, :], dim=-1)
    min_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    min_dist = dist[batch_idxs, ts_idxs, min_idxs]
    loss = min_dist
    safe_idx = loss > dis_thresh
    unsafe_idx = loss <= dis_thresh
    loss[safe_idx] = 0
    loss[unsafe_idx] = dis_thresh - loss[unsafe_idx]

    for i in range(len(intersect_idx)):
        loss[intersect_idx[i, 0], intersect_idx[i, 1] :] = 0

    return loss


class PlanMapBoundLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        map_thresh=0.5,
        lane_bound_cls_idx=2,
        dis_thresh=1.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        perception_detach=False,
    ):
        super(PlanMapBoundLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range
        self.perception_detach = perception_detach

    def forward(
        self,
        ego_fut_preds,
        lane_preds,
        lane_score_preds,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.perception_detach:
            lane_preds = lane_preds.detach()
            lane_score_preds = lane_score_preds.detach()

        # filter lane element according to confidence score and class
        not_lane_bound_mask = (
            lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        )
        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[..., 0:1] = (
            lane_bound_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        lane_bound_preds[..., 1:2] = (
            lane_bound_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_bound_loss(
            ego_fut_preds,
            lane_bound_preds,
            weight=weight,
            dis_thresh=self.dis_thresh,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_bbox


@weighted_loss
def plan_col_loss(
    pred, target, agent_fut_preds, x_dis_thresh=1.5, y_dis_thresh=3.0, dis_thresh=3.0
):
    pred = pred.cumsum(dim=-2)
    agent_fut_preds = agent_fut_preds.cumsum(dim=-2)
    target = target[:, :, None, :] + agent_fut_preds
    # filter distant agents from ego vehicle
    dist = torch.linalg.norm(pred[:, None, :, :] - target, dim=-1)
    dist_mask = dist > dis_thresh
    target[dist_mask] = 1e6

    # [B, num_agent, fut_ts]
    x_dist = torch.abs(pred[:, None, :, 0] - target[..., 0])
    y_dist = torch.abs(pred[:, None, :, 1] - target[..., 1])
    x_min_idxs = torch.argmin(x_dist, dim=1).tolist()
    y_min_idxs = torch.argmin(y_dist, dim=1).tolist()
    batch_idxs = [[i] for i in range(y_dist.shape[0])]
    ts_idxs = [[i for i in range(y_dist.shape[-1])] for j in range(y_dist.shape[0])]

    # [B, fut_ts]
    x_min_dist = x_dist[batch_idxs, x_min_idxs, ts_idxs]
    y_min_dist = y_dist[batch_idxs, y_min_idxs, ts_idxs]
    x_loss = x_min_dist
    safe_idx = x_loss > x_dis_thresh
    unsafe_idx = x_loss <= x_dis_thresh
    x_loss[safe_idx] = 0
    x_loss[unsafe_idx] = x_dis_thresh - x_loss[unsafe_idx]
    y_loss = y_min_dist
    safe_idx = y_loss > y_dis_thresh
    unsafe_idx = y_loss <= y_dis_thresh
    y_loss[safe_idx] = 0
    y_loss[unsafe_idx] = y_dis_thresh - y_loss[unsafe_idx]
    loss = torch.cat([x_loss.unsqueeze(-1), y_loss.unsqueeze(-1)], dim=-1)

    return loss


class PlanCollisionLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        agent_thresh=0.5,
        x_dis_thresh=1.5,
        y_dis_thresh=3.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
    ):
        super(PlanCollisionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.agent_thresh = agent_thresh
        self.x_dis_thresh = x_dis_thresh
        self.y_dis_thresh = y_dis_thresh
        self.pc_range = point_cloud_range

    def forward(
        self,
        ego_fut_preds,
        agent_preds,
        agent_fut_preds,
        agent_score_preds,
        agent_fut_cls_preds,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        # filter agent element according to confidence score
        agent_max_score_preds, agent_max_score_idxs = agent_score_preds.max(dim=-1)
        not_valid_agent_mask = agent_max_score_preds < self.agent_thresh
        # filter low confidence preds
        agent_fut_preds[not_valid_agent_mask] = 1e6
        # filter not vehicle preds
        not_veh_pred_mask = agent_max_score_idxs > 4  # veh idxs are 0-4
        agent_fut_preds[not_veh_pred_mask] = 1e6
        # only use best mode pred
        best_mode_idxs = torch.argmax(agent_fut_cls_preds, dim=-1).tolist()
        batch_idxs = [[i] for i in range(agent_fut_cls_preds.shape[0])]
        agent_num_idxs = [
            [i for i in range(agent_fut_cls_preds.shape[1])]
            for j in range(agent_fut_cls_preds.shape[0])
        ]
        agent_fut_preds = agent_fut_preds[batch_idxs, agent_num_idxs, best_mode_idxs]

        loss_bbox = self.loss_weight * plan_col_loss(
            ego_fut_preds,
            agent_preds,
            agent_fut_preds=agent_fut_preds,
            weight=weight,
            x_dis_thresh=self.x_dis_thresh,
            y_dis_thresh=self.y_dis_thresh,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_bbox


@weighted_loss
def plan_map_dir_loss(pred, target, dis_thresh=2.0):
    num_map_pts = target.shape[2]
    pred = pred.cumsum(dim=-2)
    traj_dis = torch.linalg.norm(pred[:, -1, :] - pred[:, 0, :], dim=-1)
    static_mask = traj_dis < 1.0
    target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)

    # find the closest map instance for ego at each timestamp
    dist = torch.linalg.norm(pred[:, :, None, None, :] - target, dim=-1)
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    target_map_inst = target[
        batch_idxs, ts_idxs, min_inst_idxs
    ]  # [B, fut_ts, num_pts, 2]

    # calculate distance
    dist = torch.linalg.norm(pred[:, :, None, :] - target_map_inst, dim=-1)
    min_pts_idxs = torch.argmin(dist, dim=-1)
    min_pts_next_idxs = min_pts_idxs.clone()
    is_end_point = min_pts_next_idxs == num_map_pts - 1
    not_end_point = min_pts_next_idxs != num_map_pts - 1
    min_pts_next_idxs[is_end_point] = num_map_pts - 2
    min_pts_next_idxs[not_end_point] = min_pts_next_idxs[not_end_point] + 1
    min_pts_idxs = min_pts_idxs.tolist()
    min_pts_next_idxs = min_pts_next_idxs.tolist()
    traj_yaw = torch.atan2(
        torch.diff(pred[..., 1]), torch.diff(pred[..., 0])
    )  # [B, fut_ts-1]
    # last ts yaw assume same as previous
    traj_yaw = torch.cat([traj_yaw, traj_yaw[:, [-1]]], dim=-1)  # [B, fut_ts]
    min_pts = target_map_inst[batch_idxs, ts_idxs, min_pts_idxs]
    dist = torch.linalg.norm(min_pts - pred, dim=-1)
    dist_mask = dist > dis_thresh
    min_pts = min_pts.unsqueeze(2)
    min_pts_next = target_map_inst[batch_idxs, ts_idxs, min_pts_next_idxs].unsqueeze(2)
    map_pts = torch.cat([min_pts, min_pts_next], dim=2)
    lane_yaw = torch.atan2(
        torch.diff(map_pts[..., 1]).squeeze(-1), torch.diff(map_pts[..., 0]).squeeze(-1)
    )  # [B, fut_ts]
    yaw_diff = traj_yaw - lane_yaw
    yaw_diff[yaw_diff > math.pi] = yaw_diff[yaw_diff > math.pi] - math.pi
    yaw_diff[yaw_diff > math.pi / 2] = yaw_diff[yaw_diff > math.pi / 2] - math.pi
    yaw_diff[yaw_diff < -math.pi] = yaw_diff[yaw_diff < -math.pi] + math.pi
    yaw_diff[yaw_diff < -math.pi / 2] = yaw_diff[yaw_diff < -math.pi / 2] + math.pi
    yaw_diff[dist_mask] = 0  # loss = 0 if no lane around ego
    yaw_diff[static_mask] = 0  # loss = 0 if ego is static

    loss = torch.abs(yaw_diff)

    return loss  # [B, fut_ts]


class PlanMapDirectionLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
        map_thresh=0.5,
        dis_thresh=2.0,
        lane_div_cls_idx=0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
    ):
        super(PlanMapDirectionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.lane_div_cls_idx = lane_div_cls_idx
        self.pc_range = point_cloud_range

    def forward(
        self,
        ego_fut_preds,
        lane_preds,
        lane_score_preds,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        # filter lane element according to confidence score and class
        not_lane_div_mask = (
            lane_score_preds[..., self.lane_div_cls_idx] < self.map_thresh
        )
        # denormalize map pts
        lane_div_preds = lane_preds.clone()
        lane_div_preds[..., 0:1] = (
            lane_div_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0])
            + self.pc_range[0]
        )
        lane_div_preds[..., 1:2] = (
            lane_div_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1])
            + self.pc_range[1]
        )
        # pad not-lane-divider cls and low confidence preds
        lane_div_preds[not_lane_div_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_dir_loss(
            ego_fut_preds,
            lane_div_preds,
            weight=weight,
            dis_thresh=self.dis_thresh,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_bbox


class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Set padding value based on boolean flag
        padding_value = 1 if self.padding else 0
        self.conv = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding_value,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


@NECKS.register_module()
class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs="on_output",
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super().__init__()
        # assert isinstance(in_channels, list)
        self.in_channels = [2048]
        self.out_channels = 256
        self.num_ins = 1
        self.num_outs = 1
        self.relu_before_extra_convs = True
        self.no_norm_on_lateral = False
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert self.num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                self.in_channels[i], self.out_channels, 1, stride=1, padding=False
            )
            fpn_conv = ConvModule(
                self.out_channels, self.out_channels, 3, stride=1, padding=True
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = self.out_channels
                extra_fpn_conv = ConvModule(
                    in_channels, self.out_channels, 3, stride=2, padding=True
                )
                self.fpn_convs.append(extra_fpn_conv)

    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.0, bias=True
        )

        self.proj_drop = nn.Dropout(p=proj_drop, inplace=False)
        self.dropout_layer = nn.Dropout(p=0.1, inplace=False)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is"
                        f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class FFN(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(FFN, self).__init__()
        self.activate = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(256, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1, inplace=False),
            ),
            nn.Linear(512, 256, bias=True),
            nn.Dropout(p=0.1, inplace=False),
        )
        self.dropout_layer = nn.Identity()

    def forward(self, x, identity=None):
        x2 = self.layers(x)
        x3 = self.dropout_layer(x2)
        return x3 + x


class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(BaseTransformerLayer, self).__init__()
        self.attentions = nn.ModuleList([MultiheadAttention()])
        self.ffns = nn.ModuleList([FFN()])

        self.norms = nn.ModuleList(
            [
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        x = self.attentions[0](
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_pos=None,
            attn_mask=None,
            key_padding_mask=None,
            **kwargs,
        )
        x = self.norms[0](x)
        x = self.ffns[0](x)
        x = self.norms[1](x)
        return x


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):

        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class CustomMSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.sampling_offsets = nn.Linear(in_features=256, out_features=64, bias=True)
        self.attention_weights = nn.Linear(in_features=256, out_features=32, bias=True)
        self.value_proj = nn.Linear(in_features=256, out_features=256, bias=True)
        self.output_proj = nn.Linear(in_features=256, out_features=256, bias=True)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 512)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=512)
        xavier_init(self.value_proj, distribution="uniform", bias=512)
        xavier_init(self.output_proj, distribution="uniform", bias=512)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """ "Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


def get_traj_warmup_loss_weight(
    cur_epoch, tot_epoch, start_pos=0.3, end_pos=0.35, scale_weight=1.1
):
    epoch_percentage = cur_epoch / tot_epoch
    sigmoid_input = 5 / (end_pos - start_pos) * epoch_percentage - 2.5 * (
        end_pos + start_pos
    ) / (end_pos - start_pos)

    return scale_weight * torch.sigmoid(torch.tensor(sigmoid_input))


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


class DetrTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super().__init__()
        self.attentions = nn.ModuleList(
            [MultiheadAttention(), CustomMSDeformableAttention()]
        )
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
            ]
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        reference_points=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        key = key if key is not None else kwargs.get("key")
        value = value if value is not None else kwargs.get("value")
        query_pos = query_pos if query_pos is not None else kwargs.get("query_pos")
        u = self.attentions[0](
            query,
            key=key,
            value=None,
            identity=None,
            query_pos=None,
            key_pos=None,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        u = self.norms[0](u)
        i = self.attentions[1](
            u,
            key=key,
            value=value,
            identity=None,
            query_pos=None,
            key_pos=None,
            attn_mask=None,
            reference_points=reference_points,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        i = self.norms[1](i)
        x = self.ffns[0](i)
        x = self.norms[2](x)
        return x


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        value_l_ = value_l_.float()
        sampling_grid_l_ = sampling_grid_l_.float()
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = 1
        self.num_heads = 8
        self.num_points = 4
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(in_features=512, out_features=128, bias=True)
        self.attention_weights = nn.Linear(in_features=512, out_features=64, bias=True)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )

        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = output.permute(1, 2, 0)

        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class MSDeformableAttention3D(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = 1
        self.num_heads = 8
        self.num_points = 8
        self.sampling_offsets = nn.Linear(in_features=256, out_features=128, bias=True)
        self.attention_weights = nn.Linear(in_features=256, out_features=64, bias=True)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_Z_anchors,
                num_Z_anchors,
                xy,
            )
            sampling_locations = reference_points + sampling_offsets
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_points,
                num_Z_anchors,
                xy,
            ) = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=256, num_levels=4
        ),
        **kwargs,
    ):
        super().__init__()
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.fp16_enabled = False
        self.deformable_attention = MSDeformableAttention3D()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2]
        )

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[
                    j, index_query_per_img
                ]
                reference_points_rebatch[
                    j, i, : len(index_query_per_img)
                ] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2
            ),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[
                    j, i, : len(index_query_per_img)
                ]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots.float())

        return self.dropout(slots) + inp_residual


class BEVFormerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList(
            [TemporalSelfAttention(), SpatialCrossAttention()]
        )
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
                nn.LayerNorm((256,), eps=1e-5, elementwise_affine=True),
            ]
        )
        self.num_attn = 2
        self.pre_norm = False

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )
        x = self.attentions[0](
            query,
            prev_bev,
            prev_bev,
            identity if self.pre_norm else None,
            query_pos=bev_pos,
            key_pos=bev_pos,
            attn_mask=attn_masks[attn_index],
            key_padding_mask=query_key_padding_mask,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )
        x = self.norms[0](x)
        x = self.attentions[1](
            x,
            key,
            value,
            identity if self.pre_norm else None,
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            mask=mask,
            attn_mask=attn_masks[attn_index],
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )
        x = self.norms[1](x)
        x = self.ffns[0](x)
        x = self.norms[2](x)
        return x


class CustomNMSFreeCoder:
    """Bbox coder for NMS-free detector (for 3D objects).
    Args:
        pc_range (list[float]): Range of point cloud, length 6 [x_min, y_min, z_min, x_max, y_max, z_max].
        post_center_range (list[float]): Limit of the center, length 6. Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score. Default: None.
        num_classes (int): Number of classes. Default: 10
    """

    def __init__(
        self, pc_range, post_center_range, max_num, score_threshold, num_classes
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, traj_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Shape [num_query, cls_out_channels].
            bbox_preds (Tensor): Shape [num_query, 9 or 10].
            traj_preds (Tensor): Shape [num_query, ...].
        Returns:
            dict: Decoded boxes.
        """
        num_query = bbox_preds.size(0)
        max_num = min(self.max_num, num_query)

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)

        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_index = bbox_index.clamp(max=num_query - 1)

        bbox_preds = bbox_preds[bbox_index]
        traj_preds = traj_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)

        final_scores = scores
        final_preds = labels
        final_traj_preds = traj_preds

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            trajs = final_traj_preds[mask]

            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
                "trajs": trajs,
            }
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            preds_dicts (dict): Contains 'all_cls_scores', 'all_bbox_preds', 'all_traj_preds'.
        Returns:
            list[dict]: Decoded boxes.
        """

        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_traj_preds = preds_dicts["all_traj_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_traj_preds[i]
                )
            )
        return predictions_list


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


class MapNMSFreeCoder:
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud, length 6 [x_min, y_min, z_min, x_max, y_max, z_max].
        post_center_range (list[float]): Limit of the center, length 8 [x1_min, y1_min, x2_min, y2_min, x1_max, y1_max, x2_max, y2_max].
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score. Default: None.
        num_classes (int): Number of classes. Default: 10
    """

    def __init__(
        self,
        pc_range,
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        num_classes=10,
    ):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l). \
                Shape [num_query, 4].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            dict: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)  # num_q,num_p,2
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :4] >= self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <= self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "map_bboxes": boxes3d,
                "map_scores": scores,
                "map_labels": labels,
                "map_pts": pts,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            preds_dicts (dict): Contains 'map_all_cls_scores', 'map_all_bbox_preds', 'map_all_pts_preds'.
        Returns:
            list[dict]: Decoded boxes.
        """

        all_cls_scores = preds_dicts["map_all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["map_all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["map_all_pts_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i]
                )
            )
        return predictions_list
