# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
import torch.nn as nn
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet.core.bbox import build_bbox_coder
from mmcv.runner import force_fp32
from mmcv.cnn import Linear, bias_init_with_prob
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import multi_apply, multi_apply


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


@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

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
        num_vec=20,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        query_embed_type="all_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        loss_pts=dict(type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
        **kwargs
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
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

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        super(MapTRHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
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
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims
                )
            else:
                self.bev_embedding = None
            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(
                    self.num_vec, self.embed_dims * 2
                )
                self.pts_embedding = nn.Embedding(
                    self.num_pts_per_vec, self.embed_dims * 2
                )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @force_fp32(apply_to=("mlvl_feats", "prev_bev"))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros(
                (bs, self.bev_h, self.bev_w), device=bev_queries.device
            ).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if (
            only_bev
        ):  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
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
                lidar_feat,
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
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](
                hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            )
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()  # cx,cy,w,h
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

        return outs

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        pts_pred,
        gt_labels,
        gt_bboxes,
        gt_shifts_pts,
        gt_bboxes_ignore=None,
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result, order_index = self.assigner.assign(
            bbox_pred,
            cls_score,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_shifts_pts,
            gt_bboxes_ignore,
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets

        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[
                sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds
            ]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[
            sampling_result.pos_assigned_gt_inds, assigned_shift, :, :
        ]
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pts_targets,
            pts_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list
