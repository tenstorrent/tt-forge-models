# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
from detectron2.layers import Conv2d, cat, get_norm

import torch
import torch.nn as nn
from .transformer import PerceptionTransformer, PerceptionTransformerV2
from detectron2.layers import ShapeSpec


class BBoxTestMixin(object):
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        print("Check1")
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Module):
    def __init__(self, init_value=0.0):
        super(Offset, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)


class AnchorFreeHead(BaseDenseHead, BBoxTestMixin):

    _version = 1

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False


class LearnedPositionalEncoding(BaseModule):
    def __init__(
        self,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (
                    x_embed.unsqueeze(0).repeat(h, 1, 1),
                    y_embed.unsqueeze(1).repeat(1, w, 1),
                ),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


class FCOS2DHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_shape,
        num_cls_convs=4,
        num_box_convs=4,
        norm="BN",
        use_deformable=False,
        use_scale=True,
        box2d_scale_init_factor=1.0,
        version="v2",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self.use_scale = use_scale
        self.box2d_scale_init_factor = box2d_scale_init_factor

        self._version = version

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if use_deformable:
            raise ValueError("Not supported yet.")

        head_configs = {"cls": num_cls_convs, "box2d": num_box_convs}

        for head_name, num_convs in head_configs.items():
            tower = []
            if self._version == "v1":
                for _ in range(num_convs):
                    conv_func = nn.Conv2d
                    tower.append(
                        conv_func(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        )
                    )
                    if norm == "GN":
                        raise NotImplementedError()
                    elif norm == "NaiveGN":
                        raise NotImplementedError()
                    elif norm == "BN":
                        tower.append(
                            ModuleListDial(
                                [
                                    nn.BatchNorm2d(in_channels)
                                    for _ in range(self.num_levels)
                                ]
                            )
                        )
                    elif norm == "SyncBN":
                        raise NotImplementedError()
                    tower.append(nn.ReLU())
            elif self._version == "v2":
                for _ in range(num_convs):
                    if norm in ("BN", "FrozenBN", "SyncBN", "GN"):
                        # NOTE: need to add norm here!
                        # Each FPN level has its own batchnorm layer.
                        # NOTE: do not use dd3d train.py!
                        # "BN" is converted to "SyncBN" in distributed training (see train.py)
                        norm_layer = ModuleListDial(
                            [
                                get_norm(norm, in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    else:
                        norm_layer = get_norm(norm, in_channels)
                    tower.append(
                        Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm_layer is None,
                            norm=norm_layer,
                            activation=F.relu,
                        )
                    )
            else:
                raise ValueError(f"Invalid FCOS2D version: {self._version}")
            self.add_module(f"{head_name}_tower", nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1
        )
        self.box2d_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.use_scale:
            if self._version == "v1":
                self.scales_reg = nn.ModuleList(
                    [
                        Scale(init_value=stride * self.box2d_scale_init_factor)
                        for stride in self.in_strides
                    ]
                )
            else:
                self.scales_box2d_reg = nn.ModuleList(
                    [
                        Scale(init_value=stride * self.box2d_scale_init_factor)
                        for stride in self.in_strides
                    ]
                )

        self.init_weights()

    def init_weights(self):

        for tower in [self.cls_tower, self.box2d_tower]:
            for l in tower.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(
                        l.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        predictors = [self.cls_logits, self.box2d_reg, self.centerness]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        box2d_reg = []
        centerness = []

        extra_output = {"cls_tower_out": []}

        for l, feature in enumerate(x):
            cls_tower_out = self.cls_tower(feature)
            bbox_tower_out = self.box2d_tower(feature)

            # 2D box
            logits.append(self.cls_logits(cls_tower_out))
            centerness.append(self.centerness(bbox_tower_out))
            box_reg = self.box2d_reg(bbox_tower_out)
            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                if self._version == "v1":
                    box_reg = self.scales_reg[l](box_reg)
                else:
                    box_reg = self.scales_box2d_reg[l](box_reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            box2d_reg.append(F.relu(box_reg))

            extra_output["cls_tower_out"].append(cls_tower_out)

        return logits, box2d_reg, centerness, extra_output


class FCOS2DLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        loc_loss_type="giou",
    ):
        super().__init__()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        # self.box2d_reg_loss_fn = IOULoss(loc_loss_type)

        self.num_classes = num_classes

    def forward(self, logits, box2d_reg, centerness, targets):
        labels = targets["labels"]
        box2d_reg_targets = targets["box2d_reg_targets"]
        pos_inds = targets["pos_inds"]

        if len(labels) != box2d_reg_targets.shape[0]:
            raise ValueError(
                f"The size of 'labels' and 'box2d_reg_targets' does not match: a={len(labels)}, b={box2d_reg_targets.shape[0]}"
            )

        # Flatten predictions
        logits = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits]
        )
        box2d_reg_pred = cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in box2d_reg])
        centerness_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) for x in centerness])

        # -------------------
        # Classification loss
        # -------------------
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        cls_target = torch.zeros_like(logits)
        cls_target[pos_inds, labels[pos_inds]] = 1

        loss_cls = (
            sigmoid_focal_loss(
                logits,
                cls_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / num_pos_avg
        )

        # NOTE: The rest of losses only consider foreground pixels.
        box2d_reg_pred = box2d_reg_pred[pos_inds]
        box2d_reg_targets = box2d_reg_targets[pos_inds]

        centerness_pred = centerness_pred[pos_inds]

        # Compute centerness targets here using 2D regression targets of foreground pixels.
        centerness_targets = compute_ctrness_targets(box2d_reg_targets)

        # Denominator for all foreground losses.
        ctrness_targets_sum = centerness_targets.sum()
        loss_denom = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        # NOTE: change the return after reduce_sum
        if pos_inds.numel() == 0:
            losses = {
                "loss_cls": loss_cls,
                "loss_box2d_reg": box2d_reg_pred.sum() * 0.0,
                "loss_centerness": centerness_pred.sum() * 0.0,
            }
            return losses, {}

        # ----------------------
        # 2D box regression loss
        # ----------------------
        loss_box2d_reg = (
            self.box2d_reg_loss_fn(
                box2d_reg_pred, box2d_reg_targets, centerness_targets
            )
            / loss_denom
        )

        # ---------------
        # Centerness loss
        # ---------------
        loss_centerness = (
            F.binary_cross_entropy_with_logits(
                centerness_pred, centerness_targets, reduction="sum"
            )
            / num_pos_avg
        )

        loss_dict = {
            "loss_cls": loss_cls,
            "loss_box2d_reg": loss_box2d_reg,
            "loss_centerness": loss_centerness,
        }
        extra_info = {
            "loss_denom": loss_denom,
            "centerness_targets": centerness_targets,
        }

        return loss_dict, extra_info


class FCOS3DHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_shape,
        num_convs=4,
        norm="BN",
        use_scale=True,
        depth_scale_init_factor=0.3,
        proj_ctr_scale_init_factor=1.0,
        use_per_level_predictors=False,
        class_agnostic=False,
        use_deformable=False,
        mean_depth_per_level=None,
        std_depth_per_level=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self.use_scale = use_scale
        self.depth_scale_init_factor = depth_scale_init_factor
        self.proj_ctr_scale_init_factor = proj_ctr_scale_init_factor
        self.use_per_level_predictors = use_per_level_predictors

        self.register_buffer("mean_depth_per_level", torch.Tensor(mean_depth_per_level))
        self.register_buffer("std_depth_per_level", torch.Tensor(std_depth_per_level))

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if use_deformable:
            raise ValueError("Not supported yet.")

        box3d_tower = []
        for i in range(num_convs):
            if norm in ("BN", "FrozenBN", "SyncBN", "GN"):
                # NOTE: need to add norm here!
                # Each FPN level has its own batchnorm layer.
                # NOTE: do not use dd3d train.py!
                # "BN" is converted to "SyncBN" in distributed training (see train.py)
                norm_layer = ModuleListDial(
                    [get_norm(norm, in_channels) for _ in range(self.num_levels)]
                )
            else:
                norm_layer = get_norm(norm, in_channels)
            box3d_tower.append(
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm_layer is None,
                    norm=norm_layer,
                    activation=F.relu,
                )
            )
        self.add_module("box3d_tower", nn.Sequential(*box3d_tower))

        num_classes = self.num_classes if not class_agnostic else 1
        num_levels = self.num_levels if use_per_level_predictors else 1

        # 3D box branches.
        self.box3d_quat = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    4 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_ctr = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    2 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_depth = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    1 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(not self.use_scale),
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_size = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    3 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_conf = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    1 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )

        if self.use_scale:
            self.scales_proj_ctr = nn.ModuleList(
                [
                    Scale(init_value=stride * self.proj_ctr_scale_init_factor)
                    for stride in self.in_strides
                ]
            )
            # (pre-)compute (mean, std) of depth for each level, and determine the init value here.
            self.scales_size = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )
            self.scales_conf = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )

            self.scales_depth = nn.ModuleList(
                [
                    Scale(init_value=sigma * self.depth_scale_init_factor)
                    for sigma in self.std_depth_per_level
                ]
            )
            self.offsets_depth = nn.ModuleList(
                [Offset(init_value=b) for b in self.mean_depth_per_level]
            )

        self._init_weights()

    def _init_weights(self):

        for l in self.box3d_tower.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    l.weight, mode="fan_out", nonlinearity="relu"
                )
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

        predictors = [
            self.box3d_quat,
            self.box3d_ctr,
            self.box3d_depth,
            self.box3d_size,
            self.box3d_conf,
        ]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf = [], [], [], [], []
        dense_depth = None
        for l, features in enumerate(x):
            box3d_tower_out = self.box3d_tower(features)

            _l = l if self.use_per_level_predictors else 0

            # 3D box
            quat = self.box3d_quat[_l](box3d_tower_out)
            proj_ctr = self.box3d_ctr[_l](box3d_tower_out)
            depth = self.box3d_depth[_l](box3d_tower_out)
            size3d = self.box3d_size[_l](box3d_tower_out)
            conf3d = self.box3d_conf[_l](box3d_tower_out)

            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                proj_ctr = self.scales_proj_ctr[l](proj_ctr)
                size3d = self.scales_size[l](size3d)
                conf3d = self.scales_conf[l](conf3d)
                depth = self.offsets_depth[l](self.scales_depth[l](depth))

            box3d_quat.append(quat)
            box3d_ctr.append(proj_ctr)
            box3d_depth.append(depth)
            box3d_size.append(size3d)
            box3d_conf.append(conf3d)

        return box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth


class FCOS3DLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        min_depth=0.1,
        max_depth=80.0,
        box3d_loss_weight=2.0,
        conf3d_loss_weight=1.0,
        conf_3d_temperature=1.0,
        smooth_l1_loss_beta=0.05,
        max_loss_per_group=20,
        predict_allocentric_rot=True,
        scale_depth_by_focal_lengths=True,
        scale_depth_by_focal_lengths_factor=500.0,
        class_agnostic=False,
        predict_distance=False,
        canon_box_sizes=None,
    ):
        super().__init__()
        self.canon_box_sizes = canon_box_sizes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.predict_allocentric_rot = predict_allocentric_rot
        self.scale_depth_by_focal_lengths = scale_depth_by_focal_lengths
        self.scale_depth_by_focal_lengths_factor = scale_depth_by_focal_lengths_factor
        self.predict_distance = predict_distance

        # self.box3d_reg_loss_fn = DisentangledBox3DLoss(smooth_l1_loss_beta, max_loss_per_group)
        self.box3d_loss_weight = box3d_loss_weight
        self.conf3d_loss_weight = conf3d_loss_weight
        self.conf_3d_temperature = conf_3d_temperature

        self.num_classes = num_classes
        self.class_agnostic = class_agnostic

    def forward(
        self,
        box3d_quat,
        box3d_ctr,
        box3d_depth,
        box3d_size,
        box3d_conf,
        dense_depth,
        inv_intrinsics,
        fcos2d_info,
        targets,
    ):
        labels = targets["labels"]
        box3d_targets = targets["box3d_targets"]
        pos_inds = targets["pos_inds"]

        if pos_inds.numel() == 0:
            losses = {
                "loss_box3d_quat": torch.stack(
                    [x.sum() * 0.0 for x in box3d_quat]
                ).sum(),
                "loss_box3d_proj_ctr": torch.stack(
                    [x.sum() * 0.0 for x in box3d_ctr]
                ).sum(),
                "loss_box3d_depth": torch.stack(
                    [x.sum() * 0.0 for x in box3d_depth]
                ).sum(),
                "loss_box3d_size": torch.stack(
                    [x.sum() * 0.0 for x in box3d_size]
                ).sum(),
                "loss_conf3d": torch.stack([x.sum() * 0.0 for x in box3d_conf]).sum(),
            }
            return losses

        if len(labels) != len(box3d_targets):
            raise ValueError(
                f"The size of 'labels' and 'box3d_targets' does not match: a={len(labels)}, b={len(box3d_targets)}"
            )

        num_classes = self.num_classes if not self.class_agnostic else 1

        box3d_quat_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4, num_classes) for x in box3d_quat]
        )
        box3d_ctr_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 2, num_classes) for x in box3d_ctr]
        )
        box3d_depth_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, num_classes) for x in box3d_depth]
        )
        box3d_size_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 3, num_classes) for x in box3d_size]
        )
        box3d_conf_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, num_classes) for x in box3d_conf]
        )

        # ----------------------
        # 3D box disentangled loss
        # ----------------------
        box3d_targets = box3d_targets[pos_inds]

        box3d_quat_pred = box3d_quat_pred[pos_inds]
        box3d_ctr_pred = box3d_ctr_pred[pos_inds]
        box3d_depth_pred = box3d_depth_pred[pos_inds]
        box3d_size_pred = box3d_size_pred[pos_inds]
        box3d_conf_pred = box3d_conf_pred[pos_inds]

        if self.class_agnostic:
            box3d_quat_pred = box3d_quat_pred.squeeze(-1)
            box3d_ctr_pred = box3d_ctr_pred.squeeze(-1)
            box3d_depth_pred = box3d_depth_pred.squeeze(-1)
            box3d_size_pred = box3d_size_pred.squeeze(-1)
            box3d_conf_pred = box3d_conf_pred.squeeze(-1)
        else:
            I = labels[pos_inds][..., None, None]
            box3d_quat_pred = torch.gather(
                box3d_quat_pred, dim=2, index=I.repeat(1, 4, 1)
            ).squeeze(-1)
            box3d_ctr_pred = torch.gather(
                box3d_ctr_pred, dim=2, index=I.repeat(1, 2, 1)
            ).squeeze(-1)
            box3d_depth_pred = torch.gather(
                box3d_depth_pred, dim=1, index=I.squeeze(-1)
            ).squeeze(-1)
            box3d_size_pred = torch.gather(
                box3d_size_pred, dim=2, index=I.repeat(1, 3, 1)
            ).squeeze(-1)
            box3d_conf_pred = torch.gather(
                box3d_conf_pred, dim=1, index=I.squeeze(-1)
            ).squeeze(-1)

        canon_box_sizes = box3d_quat_pred.new_tensor(self.canon_box_sizes)[
            labels[pos_inds]
        ]

        locations = targets["locations"][pos_inds]
        im_inds = targets["im_inds"][pos_inds]
        inv_intrinsics = inv_intrinsics[im_inds]

        box3d_pred = predictions_to_boxes3d(
            box3d_quat_pred,
            box3d_ctr_pred,
            box3d_depth_pred,
            box3d_size_pred,
            locations,
            inv_intrinsics,
            canon_box_sizes,
            self.min_depth,
            self.max_depth,
            scale_depth_by_focal_lengths_factor=self.scale_depth_by_focal_lengths_factor,
            scale_depth_by_focal_lengths=self.scale_depth_by_focal_lengths,
            quat_is_allocentric=self.predict_allocentric_rot,
            depth_is_distance=self.predict_distance,
        )

        centerness_targets = fcos2d_info["centerness_targets"]
        loss_denom = fcos2d_info["loss_denom"]
        losses_box3d, box3d_l1_error = self.box3d_reg_loss_fn(
            box3d_pred, box3d_targets, locations, centerness_targets
        )

        losses_box3d = {
            k: self.box3d_loss_weight * v / loss_denom for k, v in losses_box3d.items()
        }

        conf_3d_targets = torch.exp(-1.0 / self.conf_3d_temperature * box3d_l1_error)
        loss_conf3d = F.binary_cross_entropy_with_logits(
            box3d_conf_pred, conf_3d_targets, reduction="none"
        )
        loss_conf3d = (
            self.conf3d_loss_weight
            * (loss_conf3d * centerness_targets).sum()
            / loss_denom
        )

        losses = {"loss_conf3d": loss_conf3d, **losses_box3d}

        return losses


class DD3D(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        strides,
        fcos2d_cfg=dict(),
        fcos2d_loss_cfg=dict(),
        fcos3d_cfg=dict(),
        fcos3d_loss_cfg=dict(),
        target_assign_cfg=dict(),
        box3d_on=True,
        feature_locations_offset="none",
    ):
        super().__init__()
        # NOTE: do not need backbone
        # self.backbone = build_feature_extractor(cfg)
        # backbone_output_shape = self.backbone.output_shape()
        # self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())

        self.backbone_output_shape = [
            ShapeSpec(channels=in_channels, stride=s) for s in strides
        ]

        self.feature_locations_offset = feature_locations_offset

        self.fcos2d_head = FCOS2DHead(
            num_classes=num_classes,
            input_shape=self.backbone_output_shape,
            **fcos2d_cfg,
        )
        self.fcos2d_loss = FCOS2DLoss(num_classes=num_classes, **fcos2d_loss_cfg)
        # NOTE: inference later
        # self.fcos2d_inference = FCOS2DInference(cfg)

        if box3d_on:
            self.fcos3d_head = FCOS3DHead(
                num_classes=num_classes,
                input_shape=self.backbone_output_shape,
                **fcos3d_cfg,
            )
            self.fcos3d_loss = FCOS3DLoss(num_classes=num_classes, **fcos3d_loss_cfg)
            # NOTE: inference later
            # self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        # self.prepare_targets = DD3DTargetPreparer(num_classes=num_classes,
        #                                           input_shape=self.backbone_output_shape,
        #                                           box3d_on=box3d_on,
        #                                           **target_assign_cfg)

        # NOTE: inference later
        # self.postprocess_in_inference = cfg.DD3D.INFERENCE.DO_POSTPROCESS

        # self.do_nms = cfg.DD3D.INFERENCE.DO_NMS
        # self.do_bev_nms = cfg.DD3D.INFERENCE.DO_BEV_NMS
        # self.bev_nms_iou_thresh = cfg.DD3D.INFERENCE.BEV_NMS_IOU_THRESH

        # nuScenes inference aggregates detections over all 6 cameras.
        # self.nusc_sample_aggregate_in_inference = cfg.DD3D.INFERENCE.NUSC_SAMPLE_AGGREGATE
        self.num_classes = num_classes

        # NOTE: do not need normalize
        # self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        # self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    # NOTE:
    # @property
    # def device(self):
    #     return self.pixel_mean.device

    # def preprocess_image(self, x):
    #     return (x - self.pixel_mean) / self.pixel_std

    def forward(self, features, batched_inputs):
        # NOTE:
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [self.preprocess_image(x) for x in images]

        # NOTE: directly use inv_intrinsics
        # if 'intrinsics' in batched_inputs[0]:
        #     intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        # else:
        #     intrinsics = None
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)
        if "inv_intrinsics" in batched_inputs[0]:
            inv_intrinsics = [
                x["inv_intrinsics"].to(features[0].device) for x in batched_inputs
            ]
            inv_intrinsics = torch.stack(inv_intrinsics, dim=0)
        else:
            inv_intrinsics = None

        # NOTE:
        # gt_dense_depth = None
        # if 'depth' in batched_inputs[0]:
        #     gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
        #     gt_dense_depth = ImageList.from_tensors(
        #         gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
        #     )

        # NOTE: directly input feature
        # features = self.backbone(images.tensor)
        # features = [features[f] for f in self.in_features]

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(features[0].device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            (
                box3d_quat,
                box3d_ctr,
                box3d_depth,
                box3d_size,
                box3d_conf,
                dense_depth,
            ) = self.fcos3d_head(features)
        # NOTE: directly use inv_intrinsics
        # inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(
                locations, gt_instances, feature_shapes
            )
            # NOTE:
            # if gt_dense_depth is not None:
            #    training_targets.update({"dense_depth": gt_dense_depth})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(
                logits, box2d_reg, centerness, training_targets
            )
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat,
                    box3d_ctr,
                    box3d_depth,
                    box3d_size,
                    box3d_conf,
                    dense_depth,
                    inv_intrinsics,
                    fcos2d_info,
                    training_targets,
                )
                losses.update(fcos3d_loss)
            return losses

    def compute_locations(self, features):
        locations = []
        in_strides = [x.stride for x in self.backbone_output_shape]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h,
                w,
                in_strides[level],
                feature.dtype,
                feature.device,
                offset=self.feature_locations_offset,
            )
            locations.append(locations_per_level)
        return locations

    def forward_train(self, features, batched_inputs):
        self.train()
        return self.forward(features, batched_inputs)


class NuscenesDD3D(DD3D):
    def __init__(
        self,
        num_classes,
        in_channels,
        strides,
        fcos2d_cfg=dict(),
        fcos2d_loss_cfg=dict(),
        fcos3d_cfg=dict(),
        fcos3d_loss_cfg=dict(),
        target_assign_cfg=dict(),
        nusc_loss_weight=dict(),
        box3d_on=True,
        feature_locations_offset="none",
    ):
        super().__init__(
            num_classes,
            in_channels,
            strides,
            fcos2d_cfg=fcos2d_cfg,
            fcos2d_loss_cfg=fcos2d_loss_cfg,
            fcos3d_cfg=fcos3d_cfg,
            fcos3d_loss_cfg=fcos3d_loss_cfg,
            target_assign_cfg=target_assign_cfg,
            box3d_on=box3d_on,
            feature_locations_offset=feature_locations_offset,
        )

        # backbone_output_shape = self.backbone_output_shape
        # in_channels = backbone_output_shape[0].channels

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        self.attr_logits = Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.speed = Conv2d(
            in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation=F.relu,
        )

        # init weights
        for modules in [self.attr_logits, self.speed]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

        # # Re-define target preparer
        # del self.prepare_targets
        # self.prepare_targets = NuscenesDD3DTargetPreparer(num_classes=num_classes,
        #                                                   input_shape=self.backbone_output_shape,
        #                                                   box3d_on=box3d_on,
        #                                                   **target_assign_cfg)

        # self.nuscenes_loss = NuscenesLoss(**nusc_loss_weight)
        # NOTE: inference later
        # self.nuscenes_inference = NuscenesInference(cfg)

        # self.num_images_per_sample = cfg.MODEL.FCOS3D.NUSC_NUM_IMAGES_PER_SAMPLE
        # NOTE: inference later
        # self.num_images_per_sample = cfg.DD3D.NUSC.INFERENCE.NUM_IMAGES_PER_SAMPLE

        # assert self.num_images_per_sample == 6
        # assert cfg.DATALOADER.TEST.NUM_IMAGES_PER_GROUP == 6

        # NOTE: NuScenes evaluator allows max. 500 detections per sample.
        # self.max_num_dets_per_sample = cfg.DD3D.NUSC.INFERENCE.MAX_NUM_DETS_PER_SAMPLE

    def forward(self, features, batched_inputs):
        # NOTE:
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [self.preprocess_image(x) for x in images]

        # NOTE: directly use inv_intrinsics
        # if 'intrinsics' in batched_inputs[0]:
        #     intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        # else:
        #     intrinsics = None
        # images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)
        if "inv_intrinsics" in batched_inputs[0]:
            inv_intrinsics = [
                x["inv_intrinsics"].to(features[0].device) for x in batched_inputs
            ]
            inv_intrinsics = torch.stack(inv_intrinsics, dim=0)
        else:
            inv_intrinsics = None

        # NOTE:
        # gt_dense_depth = None
        # if 'depth' in batched_inputs[0]:
        #     gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
        #     gt_dense_depth = ImageList.from_tensors(
        #         gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
        #     )

        # NOTE: directly input feature
        # features = self.backbone(images.tensor)
        # features = [features[f] for f in self.in_features]

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(features[0].device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, fcos2d_extra_output = self.fcos2d_head(features)
        if not self.only_box2d:
            (
                box3d_quat,
                box3d_ctr,
                box3d_depth,
                box3d_size,
                box3d_conf,
                dense_depth,
            ) = self.fcos3d_head(features)
        # NOTE: directly use inv_intrinsics
        # inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        attr_logits, speeds = [], []
        for x in fcos2d_extra_output["cls_tower_out"]:
            attr_logits.append(self.attr_logits(x))
            speeds.append(self.speed(x))

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(
                locations, gt_instances, feature_shapes
            )
            # NOTE:
            # if gt_dense_depth is not None:
            #    training_targets.update({"dense_depth": gt_dense_depth})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(
                logits, box2d_reg, centerness, training_targets
            )
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat,
                    box3d_ctr,
                    box3d_depth,
                    box3d_size,
                    box3d_conf,
                    dense_depth,
                    inv_intrinsics,
                    fcos2d_info,
                    training_targets,
                )
                losses.update(fcos3d_loss)

            # Nuscenes loss -- attribute / speed
            nuscenes_loss = self.nuscenes_loss(
                attr_logits, speeds, fcos2d_info, training_targets
            )
            losses.update(nuscenes_loss)
            return losses


class DETRHead(AnchorFreeHead):

    _version = 2

    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get("class_weight", None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), (
                "Expected "
                "class_weight to have type float. Found "
                f"{type(class_weight)}."
            )
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get("bg_cls_weight", class_weight)
            assert isinstance(bg_cls_weight, float), (
                "Expected "
                "bg_cls_weight to have type float. Found "
                f"{type(bg_cls_weight)}."
            )
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_cls:
                loss_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.cls_out_channels = num_classes

        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        ACTIVATION_MAP = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
            "RReLU": nn.RReLU,
            "ReLU6": nn.ReLU6,
            "ELU": nn.ELU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
        }
        cfg = self.act_cfg.copy()
        act_type = cfg.pop("type")
        self.activate = ACTIVATION_MAP[act_type](**cfg)
        # self.activate = build_activation_layer(self.act_cfg)
        cfg = positional_encoding.copy()
        cfg.pop("type", None)
        self.positional_encoding = LearnedPositionalEncoding(**cfg)
        # self.positional_encoding = build_positional_encoding(positional_encoding)
        args = transformer.copy()
        args.pop("type", None)
        trans_type = transformer.get("type", "PerceptionTransformer")
        if trans_type == "PerceptionTransformerV2":
            self.transformer = PerceptionTransformerV2(**args)
        else:
            self.transformer = PerceptionTransformer(**args)
        # self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )
        self._init_layers()

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load checkpoints."""
        version = local_metadata.get("version", None)
        if (version is None or version < 2) and self.__class__ is DETRHead:
            convert_dict = {
                ".self_attn.": ".attentions.0.",
                ".ffn.": ".ffns.0.",
                ".multihead_attn.": ".attentions.1.",
                ".decoder.norm.": ".decoder.post_norm.",
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
