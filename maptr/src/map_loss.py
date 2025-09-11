# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.core.bbox.match_costs.builder import MATCH_COST


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
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
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_dir


@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
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
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


@MATCH_COST.register_module()
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.
    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
