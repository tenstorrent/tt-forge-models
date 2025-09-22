# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import mmcv
import numpy as np
import torch

# from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from third_party.tt_forge_models.bevdepth.pytorch.src.handlers import (
    JsonHandler,
    PickleHandler,
    YamlHandler,
)
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Union
import cv2
from abc import abstractmethod

__all__ = ["NuscDetDataset"]

map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


def imnormalize(img, mean, std, to_rgb=True):
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


class HardDiskBackend:
    def get(self, filepath: Union[str, Path]) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        with open(filepath, "r", encoding=encoding) as f:
            return f.read()


class FileClient:

    _backends = {
        "disk": HardDiskBackend,
    }

    _instances = {}

    def __new__(cls, backend: Optional[str] = None, **kwargs):
        backend = backend or "disk"
        if backend not in cls._backends:
            raise ValueError(
                f"Backend {backend} is not supported. Currently supported ones are {list(cls._backends.keys())}"
            )

        # use a simple instance cache keyed by backend and kwargs
        arg_key = backend
        for key, value in kwargs.items():
            arg_key += f":{key}:{value}"

        if arg_key in cls._instances:
            return cls._instances[arg_key]

        instance = super().__new__(cls)
        instance.client = cls._backends[backend](**kwargs)
        cls._instances[arg_key] = instance
        return instance

    @classmethod
    def infer_client(
        cls,
        file_client_args: Optional[dict] = None,
        uri: Optional[Union[str, Path]] = None,
    ) -> "FileClient":
        # Only disk backend is supported; ignore uri and default to disk when args are not provided
        return (
            cls(**file_client_args)
            if file_client_args is not None
            else cls(backend="disk")
        )

    def get(self, filepath: Union[str, Path]) -> bytes:
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        return self.client.get_text(filepath, encoding)


FileLikeObject = Union[TextIO, StringIO, BytesIO]

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}


def is_str(x):
    return isinstance(x, str)


def load(file, file_format=None, file_client_args=None, **kwargs):
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if is_str(file):
        file_client = FileClient.infer_client(file_client_args, file)
        if handler.str_like:
            with StringIO(file_client.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_client.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

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
        """torch.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """torch.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """torch.Tensor: A vector with height of each box."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """torch.Tensor: A vector with the top height of each box."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """torch.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In the MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for more clear usage.

        Returns:
            torch.Tensor: A tensor with center of each box.
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        pass

    @property
    def corners(self):
        """torch.Tensor: a tensor with 8 corners of each box."""
        pass

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (torch.Tensor): Translation vector of size 1x3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each box is \
                inside the reference range.
        """
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
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each box is inside \
                the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            torch.Tensor: A binary vector which represents whether each \
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = (size_x > threshold) & (size_y > threshold) & (size_z > threshold)
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of  \
                :class:`BaseInstances3DBoxes` after indexing.
        """
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
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, boxes_list):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
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
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the \
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw
        )

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw
        )

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode="iou"):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes.
        """
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
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
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
        """Create a new box object with data.

        The new box and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, torch.Tensor)
            else data.to(self.device)
        )
        original_type = type(self)
        return original_type(new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                            up z    x front (yaw=-0.5*pi)
                               ^   ^
                               |  /
                               | /
      (yaw=-pi) left y <------ 0 -------- (yaw=0)

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and decreases from
    the negative direction of y to the positive direction of x.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
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
        """torch.Tensor: 2D BEV box of each box with rotation
        in XYWHR format."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
        without rotation."""
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
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns \
                None, otherwise it returns the rotated points and the \
                rotation matrix ``rot_mat_T``.
        """
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
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
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
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Whether each box is inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 0] < box_range[2])
            & (self.tensor[:, 1] < box_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): the target Box mode
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: \
                The converted box of the same type in the ``dst`` mode.
        """
        # from .box_3d_mode import Box3DMode
        return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def points_in_boxes(self, points):
        """Find the box which the points are in.

        Args:
            points (torch.Tensor): Points in shape (N, 3).

        Returns:
            torch.Tensor: The index of box where each point are in.
        """
        box_idx = points_in_boxes_gpu(
            points.unsqueeze(0), self.tensor.unsqueeze(0).to(points.device)
        ).squeeze(0)
        return box_idx


def get_rot(h):
    return torch.Tensor(
        [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
    )


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    scale_mat = torch.Tensor(
        [[scale_ratio, 0, 0], [0, scale_ratio, 0], [0, 0, scale_ratio]]
    )
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat


def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = (
        (depth_coords[:, 1] < resize_dims[0])
        & (depth_coords[:, 0] < resize_dims[1])
        & (depth_coords[:, 1] >= 0)
        & (depth_coords[:, 0] >= 0)
    )
    depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0]] = cam_depth[
        valid_mask, 2
    ]

    return torch.Tensor(depth_map)


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor["translation"]))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose["rotation"]).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose["translation"]))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose["translation"]))
    lidar_points.rotate(Quaternion(cam_ego_pose["rotation"]).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor["translation"]))
    lidar_points.rotate(Quaternion(cam_calibrated_sensor["rotation"]).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(
        lidar_points.points[:3, :],
        np.array(cam_calibrated_sensor["camera_intrinsic"]),
        normalize=True,
    )

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


class NuscDetDataset(Dataset):
    def __init__(
        self,
        ida_aug_conf,
        bda_aug_conf,
        classes,
        data_root,
        info_paths,
        is_train,
        use_cbgs=False,
        num_sweeps=1,
        img_conf=dict(
            img_mean=[123.675, 116.28, 103.53],
            img_std=[58.395, 57.12, 57.375],
            to_rgb=True,
        ),
        return_depth=False,
        sweep_idxes=list(),
        key_idxes=list(),
        use_fusion=False,
    ):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            return_depth (bool): Whether to use depth gt.
                default: False.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
            use_fusion (bool): Whether to use lidar data.
                default: False.
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(load(info_path))
        else:
            self.infos = load(info_paths)
        self.is_train = is_train
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf["img_mean"], np.float32)
        self.img_std = np.array(img_conf["img_std"], np.float32)
        self.to_rgb = img_conf["to_rgb"]
        self.return_depth = return_depth
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) == len(
            sweep_idxes
        ), "All `sweep_idxes` must greater \
                than or equal to 0."

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(
            key_idxes
        ), "All `key_idxes` must less than 0."
        self.key_idxes = [0] + key_idxes
        self.use_fusion = use_fusion

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info["category_name"] for ann_info in info["ann_infos"]]
            )
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf["H"], self.ida_aug_conf["W"]
        fH, fW = self.ida_aug_conf["final_dim"]
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.ida_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf["rot_lim"])
            scale_bda = np.random.uniform(*self.bda_aug_conf["scale_lim"])
            flip_dx = np.random.uniform() < self.bda_aug_conf["flip_dx_ratio"]
            flip_dy = np.random.uniform() < self.bda_aug_conf["flip_dy_ratio"]
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info["LIDAR_TOP"]["calibrated_sensor"]
        lidar_ego_pose = lidar_info["LIDAR_TOP"]["ego_pose"]
        cam_calibrated_sensor = cam_info["calibrated_sensor"]
        cam_ego_pose = cam_info["ego_pose"]
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(),
            img,
            lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(),
            cam_calibrated_sensor,
            cam_ego_pose,
        )
        return np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(
            np.float32
        )

    def get_image(self, cam_infos, cams, lidar_infos=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info["LIDAR_TOP"]["filename"]
                lidar_points = np.fromfile(
                    os.path.join(self.data_root, lidar_path), dtype=np.float32, count=-1
                ).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()
            for sweep_idx, cam_info in enumerate(cam_infos):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]["filename"])
                )
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]["calibrated_sensor"]["rotation"]
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix
                )
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]["calibrated_sensor"]["translation"]
                )
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]["ego_pose"]["rotation"]
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix
                )
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]["ego_pose"]["translation"]
                )
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]["ego_pose"]["rotation"]
                keyego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]["ego_pose"]["translation"]
                )
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]["calibrated_sensor"]["rotation"]
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix
                )
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]["calibrated_sensor"]["translation"]
                )
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor
                    @ global2keyego
                    @ sweepego2global
                    @ sweepsensor2sweepego
                ).inverse()
                sweepsensor2keyego = (
                    global2keyego @ sweepego2global @ sweepsensor2sweepego
                )
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]["calibrated_sensor"]["camera_intrinsic"]
                )
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx],
                        img,
                        lidar_infos[sweep_idx],
                        cam_info[cam],
                    )
                    point_depth_augmented = depth_transform(
                        point_depth,
                        resize,
                        self.ida_aug_conf["final_dim"],
                        crop,
                        flip,
                        rotate_ida,
                    )
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = imnormalize(
                    np.array(img), self.img_mean, self.img_std, self.to_rgb
                )
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]["timestamp"])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]["ego_pose"]["rotation"] for cam in cams], 0
        )
        ego2global_translation = np.mean(
            [key_info[cam]["ego_pose"]["translation"] for cam in cams], 0
        )
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info["cam_infos"][cam]["ego_pose"]["rotation"] for cam in cams], 0
        )
        ego2global_translation = np.mean(
            [info["cam_infos"][cam]["ego_pose"]["translation"] for cam in cams], 0
        )
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info["ann_infos"]:
            # Use ego coordinate.
            if (
                map_name_from_general_to_detection[ann_info["category_name"]]
                not in self.classes
                or ann_info["num_lidar_pts"] + ann_info["num_radar_pts"] <= 0
            ):
                continue
            box = Box(
                ann_info["translation"],
                ann_info["size"],
                Quaternion(ann_info["rotation"]),
                velocity=ann_info["velocity"],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(
                    map_name_from_general_to_detection[ann_info["category_name"]]
                )
            )
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf["Ncams"] < len(
            self.ida_aug_conf["cams"]
        ):
            cams = np.random.choice(
                self.ida_aug_conf["cams"], self.ida_aug_conf["Ncams"], replace=False
            )
        else:
            cams = self.ida_aug_conf["cams"]
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]["scene_token"] != self.infos[idx]["scene_token"]:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info["cam_infos"])
            lidar_infos.append(info["lidar_infos"])
            lidar_sweep_timestamps = [
                lidar_sweep["LIDAR_TOP"]["timestamp"]
                for lidar_sweep in info["lidar_sweeps"]
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info["cam_sweeps"]) == 0:
                    cam_infos.append(info["cam_infos"])
                    lidar_infos.append(info["lidar_infos"])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info["cam_sweeps"]) - 1, sweep_idx), -1, -1):
                        if sum([cam in info["cam_sweeps"][i] for cam in cams]) == len(
                            cams
                        ):
                            cam_infos.append(info["cam_sweeps"][i])
                            cam_timestamp = np.mean(
                                [
                                    val["timestamp"]
                                    for val in info["cam_sweeps"][i].values()
                                ]
                            )
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(
                                lidar_sweep_timestamps - cam_timestamp
                            ).argmin()
                            lidar_infos.append(info["lidar_sweeps"][lidar_idx])
                            break
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)

        else:
            image_data_list = self.get_image(cam_infos, cams)
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        img_metas["token"] = self.infos[idx]["sample_token"]
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(
                0,
            )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(
            gt_boxes, rotate_bda, scale_bda, flip_dx, flip_dy
        )
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)


def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
    mats_dict = dict()
    mats_dict["sensor2ego_mats"] = torch.stack(sensor2ego_mats_batch)
    mats_dict["intrin_mats"] = torch.stack(intrin_mats_batch)
    mats_dict["ida_mats"] = torch.stack(ida_mats_batch)
    mats_dict["sensor2sensor_mats"] = torch.stack(sensor2sensor_mats_batch)
    mats_dict["bda_mat"] = torch.stack(bda_mat_batch)
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    return ret_list
