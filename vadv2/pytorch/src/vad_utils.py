# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import cv2
from skimage.draw import polygon
import copy

ego_width, ego_length = 1.85, 4.084


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


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


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
        in_channels=[2048],
        out_channels=256,
        num_outs=1,
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


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_feats=128,
        row_num_embed=100,
        col_num_embed=100,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
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

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"row_num_embed={self.row_num_embed}, "
        repr_str += f"col_num_embed={self.col_num_embed})"
        return repr_str


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit), nn.LayerNorm(hidden_unit), nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f"lmlp_{i}", MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        """
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        """
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max
