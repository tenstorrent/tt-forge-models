# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion model loader for tt-forge-models.

BEVFusion is a multi-task, multi-sensor (Camera + LiDAR) fusion framework
for 3D object detection. This loader clones the upstream repo, applies a
CPU-inference patch, builds the required extensions, and provides the
pretrained model with random inputs matching real nuScenes data shapes.
"""

import os
import sys
import subprocess
import hashlib
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel

_REPO_URL = "https://github.com/mit-han-lab/bevfusion.git"

_CHECKPOINT_URLS = {
    "camera+lidar": "https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1",
    "camera-only": "https://www.dropbox.com/scl/fi/pxfaz1nc07qa2twlatzkz/camera-only-det.pth?rlkey=f5do81fawie0ssbg9uhrm6p30&dl=1",
    "lidar-only": "https://www.dropbox.com/scl/fi/b1zvgrg9ucmv0wtx6pari/lidar-only-det.pth?rlkey=fw73bmdh57jxtudw6osloywah&dl=1",
}

_CONFIG_PATHS = {
    "camera+lidar": "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    "camera-only": "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml",
    "lidar-only": "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml",
}


def _download_file(url: str, cache_dir: str = None) -> str:
    """Download a file from URL to a local cache, return the cached path."""
    if cache_dir is None:
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "bevfusion", "weights",
        )
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    basename = os.path.basename(url.split("?")[0]) or "weights.pth"
    local_path = os.path.join(cache_dir, f"{url_hash}_{basename}")
    if not os.path.isfile(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


class ModelVariant(StrEnum):
    CAMERA_LIDAR = "camera+lidar"
    CAMERA_ONLY = "camera-only"
    LIDAR_ONLY = "lidar-only"


class ModelLoader(ForgeModel):
    """BEVFusion model loader."""

    _VARIANTS = {
        ModelVariant.CAMERA_LIDAR: ModelConfig(pretrained_model_name="bevfusion-det"),
        ModelVariant.CAMERA_ONLY: ModelConfig(pretrained_model_name="camera-only-det"),
        ModelVariant.LIDAR_ONLY: ModelConfig(pretrained_model_name="lidar-only-det"),
    }

    DEFAULT_VARIANT = ModelVariant.CAMERA_LIDAR

    _CACHE_DIR = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "bevfusion",
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._repo_dir = os.path.join(self._CACHE_DIR, "bevfusion")

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="bevfusion",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MULTIVIEW_3D_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _ensure_repo(self):
        """Clone the BEVFusion repo, apply the CPU-inference patch, and build extensions."""
        repo_exists = os.path.isdir(os.path.join(self._repo_dir, "mmdet3d"))

        if not repo_exists:
            os.makedirs(self._CACHE_DIR, exist_ok=True)

            patch_file = Path(__file__).parent / "patches" / "cpu-inference.patch"
            if not patch_file.exists():
                raise FileNotFoundError(f"CPU-inference patch not found at {patch_file}")

            subprocess.check_call(
                ["git", "clone", "--depth", "1", _REPO_URL, self._repo_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.check_call(
                ["git", "am", str(patch_file.resolve())],
                cwd=self._repo_dir,
            )

        self._build_extensions()

    def _build_extensions(self):
        """Build BEVFusion C++ extensions (CPU-only) if not already built."""
        marker = os.path.join(self._repo_dir, ".cpu_extensions_built")
        if os.path.isfile(marker):
            return

        subprocess.check_call(
            [sys.executable, "setup.py", "develop"],
            cwd=self._repo_dir,
            stdout=subprocess.DEVNULL,
        )

        with open(marker, "w") as f:
            f.write("done")

    def _add_to_path(self):
        if self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)

    @staticmethod
    def _patch_mmcv_ext_loader():
        """Patch mmcv's extension loader to handle missing C++ extensions.

        We use mmcv (lite) which lacks compiled C++ ops (mmcv._ext).
        mmdet eagerly imports mmcv.ops which triggers ext_loader.load_ext().
        Since BEVFusion has its own CPU fallbacks, we stub out the loader.
        """
        import types
        import mmcv.utils.ext_loader as ext_loader

        _original = ext_loader.load_ext

        def _safe_load_ext(name, funcs):
            try:
                return _original(name, funcs)
            except (ImportError, ModuleNotFoundError):
                dummy = types.ModuleType(name)
                for fn in funcs:
                    setattr(dummy, fn, None)
                return dummy

        ext_loader.load_ext = _safe_load_ext

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_repo()
        self._add_to_path()

        warnings.filterwarnings("ignore")
        self._patch_mmcv_ext_loader()

        from torchpack.utils.config import configs
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        from mmdet3d.utils import recursive_eval

        variant_key = self._variant.value
        config_path = os.path.join(self._repo_dir, _CONFIG_PATHS[variant_key])

        configs.load(config_path, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_path)
        cfg.model.train_cfg = None

        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        model.eval()

        ckpt_path = _download_file(_CHECKPOINT_URLS[variant_key])
        load_checkpoint(model, ckpt_path, map_location="cpu")

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, num_points=30000):
        """Load random inputs matching real nuScenes data shapes.
        Shapes match nuScenes Camera+LiDAR data:
            img:               [B, 6, 3, 256, 704]  (6 camera views)
            points:            list of [N, 5] tensors (x, y, z, intensity, ring)
            camera2ego:        [B, 6, 4, 4]
            lidar2ego:         [B, 4, 4]
            lidar2camera:      [B, 6, 4, 4]
            lidar2image:       [B, 6, 4, 4]
            camera_intrinsics: [B, 6, 4, 4]
            camera2lidar:      [B, 6, 4, 4]
            img_aug_matrix:    [B, 6, 4, 4]
            lidar_aug_matrix:  [B, 4, 4]
            metas:             list of dicts
        Args:
            dtype_override: Optional torch.dtype for float tensors.
            batch_size: Batch size (default 1).
            num_points: Number of LiDAR points per sample (default 30000).
        Returns:
            dict: Input dictionary that can be unpacked into model(**inputs)."""
        self._add_to_path()
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        dt = torch.float32
        B = batch_size
        N = 6

        img = torch.randn(B, N, 3, 256, 704, dtype=dt)

        points = []
        for _ in range(B):
            pts = torch.randn(num_points, 5, dtype=dt)
            pts[:, :3] *= 50.0
            points.append(pts)

        camera2ego = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
        lidar2ego = torch.eye(4, dtype=dt).unsqueeze(0).expand(B, -1, -1).contiguous()
        lidar2camera = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
        lidar2image = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()

        camera_intrinsics = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        camera_intrinsics[:, :, 0, 0] = 1266.4
        camera_intrinsics[:, :, 1, 1] = 1266.4
        camera_intrinsics[:, :, 0, 2] = 816.3
        camera_intrinsics[:, :, 1, 2] = 491.5

        camera2lidar = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
        img_aug_matrix = torch.eye(4, dtype=dt).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).contiguous()
        lidar_aug_matrix = torch.eye(4, dtype=dt).unsqueeze(0).expand(B, -1, -1).contiguous()

        metas = [
            {"token": f"random_sample_{i}", "box_type_3d": LiDARInstance3DBoxes}
            for i in range(B)
        ]

        return {
            "img": img,
            "points": points,
            "camera2ego": camera2ego,
            "lidar2ego": lidar2ego,
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "camera_intrinsics": camera_intrinsics,
            "camera2lidar": camera2lidar,
            "img_aug_matrix": img_aug_matrix,
            "lidar_aug_matrix": lidar_aug_matrix,
            "metas": metas,
            "depths": None,
        }
