# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numbers
import os.path as osp
import random
from abc import abstractmethod
from collections import abc
from collections.abc import Sequence
from copy import deepcopy
from enum import IntEnum, unique
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import torch
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)
from .handlers import JsonHandler, PickleHandler, YamlHandler
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from .nuscenes_dataloader import DataContainer as DC
from torch.utils.data import Dataset

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None
supported_backends = ["cv2", "turbojpeg", "pillow", "tifffile"]

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}

imread_backend = "cv2"

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}
cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

if Image is not None:
    pillow_interp_codes = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "box": Image.BOX,
        "lanczos": Image.LANCZOS,
        "hamming": Image.HAMMING,
    }

    data_test = {
        "type": "CustomNuScenesDataset",
        "data_root": "/proj_sw/user_dev/mramanathan/bgdlab19_sep10_xla/tt-xla/tests/jax/single_chip/models/bevformer/data/nuscenes",
        "ann_file": "/proj_sw/user_dev/mramanathan/bgdlab19_sep10_xla/tt-xla/tests/jax/single_chip/models/bevformer/data/nuscenes/nuscenes_infos_temporal_val.pkl",
        "pipeline": [
            {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
            {
                "type": "NormalizeMultiviewImage",
                "mean": [123.675, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": False,
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


def get_test_dataset_cfg(variant: str):
    """Return test dataset config tuned per variant: tiny/small/base."""
    if variant == "BEVFormer-small":
        bev_h = 150
        bev_w = 150
        norm = {
            "mean": [103.530, 116.280, 123.675],
            "std": [1.0, 1.0, 1.0],
            "to_rgb": False,
        }
        scale = 0.8
        use_random_scale = True
    elif variant == "BEVFormer-base":
        bev_h = 200
        bev_w = 200
        norm = {
            "mean": [103.530, 116.280, 123.675],
            "std": [1.0, 1.0, 1.0],
            "to_rgb": False,
        }
        scale = 0.8
        use_random_scale = False
    else:
        bev_h = 50
        bev_w = 50
        norm = {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "to_rgb": True,
        }
        scale = 0.5
        use_random_scale = True

    cfg = data_test.copy()

    pipeline = []
    for p in cfg["pipeline"]:
        p = p.copy()

        if p.get("type") == "NormalizeMultiviewImage":
            p.update(
                {"mean": norm["mean"], "std": norm["std"], "to_rgb": norm["to_rgb"]}
            )

        if p.get("type") == "MultiScaleFlipAug3D":
            inner = []
            for t in p.get("transforms", []):
                t = t.copy()

                if (
                    t.get("type") == "RandomScaleImageMultiViewImage"
                    and not use_random_scale
                ):
                    continue

                if (
                    t.get("type") == "RandomScaleImageMultiViewImage"
                    and use_random_scale
                ):
                    t["scales"] = [scale]

                inner.append(t)
            p["transforms"] = inner

        pipeline.append(p)

    cfg["pipeline"] = pipeline
    cfg["bev_size"] = (bev_h, bev_w)
    return cfg


def build_dataset(cfg):
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    args = cfg.copy()
    ds_type = args.pop("type")
    if ds_type == "CustomNuScenesDataset":
        return CustomNuScenesDataset(**args)
    if ds_type == "ConcatDataset":
        datasets = [build_dataset(c) for c in args["datasets"]]
        from torch.utils.data import ConcatDataset as TorchConcatDataset

        return TorchConcatDataset(datasets)
    if ds_type == "RepeatDataset":
        inner = build_dataset(args["dataset"])
        times = args["times"]
        # Simple repeat wrapper
        class _Repeat:
            def __init__(self, ds, times):
                self.ds = ds
                self.times = times
                self._len = len(ds) * times

            def __len__(self):
                return self._len

            def __getitem__(self, idx):
                return self.ds[idx % len(self.ds)]

        return _Repeat(inner, times)
    raise KeyError(f"Unsupported dataset type: {ds_type}")


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


def to_tensor(data):

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


def imresize(
    img, size, return_scale=False, interpolation="bilinear", out=None, backend=None
):
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    return is_seq_of(seq, expected_type, seq_type=list)


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


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def _pillow2array(img, flag="color", channel_order="bgr"):

    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ["color", "grayscale"]:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != "RGB":
            if img.mode != "LA":
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert("RGB")
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ["color", "color_ignore_orientation"]:
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ["grayscale", "grayscale_ignore_orientation"]:
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f" but got {flag}"
            )
    return array


def _jpegflag(flag="color", channel_order="bgr"):
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "color":
        if channel_order == "bgr":
            return TJPF_BGR
        elif channel_order == "rgb":
            return TJCS_RGB
    elif flag == "grayscale":
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def imread(img_or_path, flag="color", channel_order="bgr", backend=None):

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported "
            "backends are 'cv2', 'turbojpeg', 'pillow'"
        )
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        # Remap relative './data' path to absolute dataset path
        if img_or_path.startswith("./data"):
            base_data_root = "/proj_sw/user_dev/mramanathan/bgdlab19_sep10_xla/tt-xla/tests/jax/single_chip/models/bevformer/data"
            img_or_path = base_data_root + img_or_path[len("./data") :]
        check_file_exist(img_or_path, f"img file does not exist: {img_or_path}")
        if backend == "turbojpeg":
            with open(img_or_path, "rb") as in_file:
                img = jpeg.decode(in_file.read(), _jpegflag(flag, channel_order))
                if img.shape[-1] == 1:
                    img = img[:, :, 0]
            return img
        elif backend == "pillow":
            img = Image.open(img_or_path)
            img = _pillow2array(img, flag, channel_order)
            return img
        elif backend == "tifffile":
            img = tifffile.imread(img_or_path)
            return img
        else:
            flag = imread_flags[flag] if is_str(flag) else flag
            img = cv2.imread(img_or_path, flag)
            if flag == IMREAD_COLOR and channel_order == "rgb":
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img
    else:
        raise TypeError(
            '"img" must be a numpy array or a str or ' "a pathlib.Path object"
        )


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode="constant"):

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(
            "pad_val must be a int or a tuple. " f"But received {type(pad_val)}"
        )

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(
            "Padding must be a int or a 2, or 4 element tuple."
            f"But received {padding}"
        )

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val,
    )

    return img


def impad_to_multiple(img, divisor, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


class CustomCollect3D(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "lidar2cam",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "can_bus",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data["img_metas"] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            if key not in results:
                data[key] = None
            else:
                data[key] = results[key]
        return data

    def __repr__(self):

        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


class DefaultFormatBundle(object):
    def __init__(
        self,
    ):
        return

    def __call__(self, results):

        if "img" in results:
            if isinstance(results["img"], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results["img"] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results["img"].transpose(2, 0, 1))
                results["img"] = DC(to_tensor(img), stack=True)
        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))

        if "gt_masks" in results:
            results["gt_masks"] = DC(results["gt_masks"], cpu_only=True)
        if "gt_semantic_seg" in results:
            results["gt_semantic_seg"] = DC(
                to_tensor(results["gt_semantic_seg"][None, ...]), stack=True
            )

        return results

    def __repr__(self):
        return self.__class__.__name__


class DefaultFormatBundle3D(DefaultFormatBundle):
    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label


class PadMultiViewImage(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):

        if self.size is not None:
            padded_img = [
                impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]

        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):

        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


class RandomScaleImageMultiViewImage(object):
    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def __call__(self, results):

        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results["img"]]
        x_size = [int(img.shape[1] * rand_scale) for img in results["img"]]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results["img"] = [
            imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
            for idx, img in enumerate(results["img"])
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        results["lidar2img"] = lidar2img
        results["img_shape"] = [img.shape for img in results["img"]]
        results["ori_shape"] = [img.shape for img in results["img"]]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.scales}, "
        return repr_str


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = []
        for t in transforms:
            # If already a callable (class instance), keep it; dicts are supported by instantiating from globals
            if callable(t):
                self.transforms.append(t)
            elif isinstance(t, dict):
                cfg = t.copy()
                if "type" not in cfg:
                    raise KeyError('transform cfg must contain key "type"')
                t_type = cfg.pop("type")
                cls = globals().get(t_type)
                if cls is None:
                    raise KeyError(f"Unknown transform type: {t_type}")
                self.transforms.append(cls(**cfg))
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        parts = [self.__class__.__name__ + "("]
        for t in self.transforms:
            parts.append(f"    {t}")
        parts.append(")")
        return "\n".join(parts)


class MultiScaleFlipAug3D(object):
    def __init__(
        self,
        transforms,
        img_scale,
        pts_scale_ratio,
        flip=False,
        flip_direction="horizontal",
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
    ):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = (
            pts_scale_ratio
            if isinstance(pts_scale_ratio, list)
            else [float(pts_scale_ratio)]
        )

        assert is_list_of(self.img_scale, tuple)
        assert is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = (
            flip_direction if isinstance(flip_direction, list) else [flip_direction]
        )
        assert is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ["horizontal"]:
            warnings.warn("flip_direction has no effect when flip is set to False")
        if self.flip and not any(
            [
                (t["type"] == "RandomFlip3D" or t["type"] == "RandomFlip")
                for t in transforms
            ]
        ):
            warnings.warn("flip has no effect when RandomFlip is not in transforms")

    def __call__(self, results):

        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = (
            [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        )
        pcd_vertical_flip_aug = (
            [False, True] if self.flip and self.pcd_vertical_flip else [False]
        )
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results["scale"] = scale
                                _results["flip"] = flip
                                _results["pcd_scale_factor"] = pts_scale_ratio
                                _results["flip_direction"] = direction
                                _results["pcd_horizontal_flip"] = pcd_horizontal_flip
                                _results["pcd_vertical_flip"] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f"(transforms={self.transforms}, "
        repr_str += f"img_scale={self.img_scale}, flip={self.flip}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip_direction={self.flip_direction})"
        return repr_str


class NormalizeMultiviewImage(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):

        results["img"] = [
            imnormalize(img, self.mean, self.std, self.to_rgb) for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


class LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):

        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack([imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


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


@unique
class Box3DMode(IntEnum):

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(box, src, dst, rt_mat=None):
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                "Box3DMode.convert takes either a k-tuple/list or "
                "an Nxk array/tensor, where k >= 7"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([y_size, z_size, x_size], dim=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([z_size, x_size, y_size], dim=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([y_size, x_size, z_size], dim=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([y_size, x_size, z_size], dim=-1)
        else:
            raise NotImplementedError(
                f"Conversion from Box3DMode {src} to {dst} " "is not supported yet"
            )

        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat([arr[:, :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[:, :3] @ rt_mat.t()

        remains = arr[..., 6:]
        arr = torch.cat([xyz[:, :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(
                    f"Conversion to {dst} through {original_type}"
                    " is not supported yet"
                )
            return target_type(arr, box_dim=arr.size(-1), with_yaw=box.with_yaw)
        else:
            return arr


def get_box_type(box_type):

    # import Box3DMode, CameraInstance3DBoxes,
    #                           DepthInstance3DBoxes, LiDARInstance3DBoxes)
    box_type_lower = box_type.lower()
    if box_type_lower == "lidar":
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == "camera":
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == "depth":
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError(
            'Only "box_type" of "camera", "lidar", "depth"'
            f" are supported, got {box_type}"
        )

    return box_type_3d, box_mode_3d


# @DATASETS.register_module()
class Custom3DDataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        return load(ann_file)

    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info["point_cloud"]["lidar_idx"]
        pts_filename = osp.join(self.data_root, info["pts_path"])

        input_dict = dict(
            pts_filename=pts_filename, sample_idx=sample_idx, file_name=pts_filename
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        return input_dict

    def pre_pipeline(self, results):
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_test_data(self, index):

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):

        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def __len__(self):
        return len(self.data_infos)


class NuScenesDataset(Custom3DDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def load_annotations(self, ann_file):

        data = load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos


class CustomNuScenesDataset(NuScenesDataset):
    def __init__(
        self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
