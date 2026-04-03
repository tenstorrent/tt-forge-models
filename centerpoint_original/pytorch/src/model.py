# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterPoint model using the ORIGINAL tianweiy/CenterPoint det3d code.

Reference: https://github.com/tianweiy/CenterPoint (CVPR 2021)

Uses the patched det3d submodule from Toyota-fresh/third_party/centerpoint/
with CUDA/numba/spconv imports monkey-patched out for CPU inference.

Architecture (PointPillars variant):
  BEV pseudo-image (B, 64, 512, 512) → det3d RPN neck → det3d CenterHead → raw heatmaps

The det3d RPN has ZeroPad2d at position 0 in each block, so checkpoint indices
are shifted +1 compared to the mmdet3d checkpoint keys.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

_REPO_URL = "https://github.com/tianweiy/CenterPoint.git"
_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
) / "centerpoint_original"

_det3d_ready = False


def _find_patched_repo() -> str:
    """Locate the patched CenterPoint det3d repo.

    Search order:
      1. Toyota-fresh/third_party/centerpoint/ (workspace-local, already patched)
      2. Cached clone at ~/.cache/centerpoint_original/CenterPoint/ (auto-patched)
    """
    workspace = Path(__file__).resolve()
    for _ in range(10):
        workspace = workspace.parent
        if (workspace / "Toyota-fresh" / "third_party" / "centerpoint" / "det3d").is_dir():
            return str(workspace / "Toyota-fresh" / "third_party" / "centerpoint")
    return ""


def _get_patch_file() -> str:
    """Locate the cpu-inference.patch file in the workspace."""
    workspace = Path(__file__).resolve()
    for _ in range(10):
        workspace = workspace.parent
        patch = workspace / "Toyota-fresh" / "centerpoint" / "model" / "patches" / "cpu-inference.patch"
        if patch.exists():
            return str(patch)
    return ""


def _clone_and_patch() -> str:
    """Clone the CenterPoint repo and apply cpu-inference patch."""
    repo_dir = _CACHE_DIR / "CenterPoint"

    if not repo_dir.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--depth", "1", _REPO_URL, str(repo_dir)]
        )

    marker = repo_dir / ".cpu_inference_patched"
    if not marker.exists():
        patch_file = _get_patch_file()
        if patch_file:
            subprocess.check_call(
                ["git", "apply", "--whitespace=nowarn", patch_file],
                cwd=str(repo_dir),
            )
            marker.write_text("applied\n")
        else:
            _apply_inline_patches(str(repo_dir))
            marker.write_text("applied-inline\n")

    return str(repo_dir)


def _apply_inline_patches(repo_dir: str):
    """Apply minimal patches inline when the .patch file is not available.

    Handles the three critical import-time failures:
      1. numba imports in several files
      2. spconv import guard in models/__init__.py
      3. roi_heads ImportError guard in models/__init__.py
    """
    import re

    numba_guard = (
        'try:\n    import numba\nexcept ImportError:\n'
        '    import types as _t\n'
        '    numba = _t.SimpleNamespace(\n'
        '        njit=lambda f=None, **kw: (lambda g: g) if f is None else f,\n'
        '        jit=lambda f=None, **kw: (lambda g: g) if f is None else f,\n'
        '    )\n'
    )

    files_needing_numba = [
        "det3d/core/bbox/box_np_ops.py",
        "det3d/core/bbox/geometry.py",
        "det3d/models/utils/misc.py",
        "det3d/ops/point_cloud/bev_ops.py",
        "det3d/ops/point_cloud/point_cloud_ops.py",
        "det3d/core/utils/circle_nms_jit.py",
    ]

    for rel_path in files_needing_numba:
        fpath = os.path.join(repo_dir, rel_path)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            content = f.read()
        if "try:" not in content.split("import numba")[0][-30:] if "import numba" in content else True:
            content = content.replace("import numba\n", numba_guard)
            with open(fpath, "w") as f:
                f.write(content)

    models_init = os.path.join(repo_dir, "det3d/models/__init__.py")
    if os.path.exists(models_init):
        with open(models_init, "r") as f:
            content = f.read()
        content = content.replace(
            "from .roi_heads import *",
            "try:\n    from .roi_heads import *\nexcept ImportError:\n    pass",
        )
        with open(models_init, "w") as f:
            f.write(content)


def _setup_det3d():
    """Ensure det3d is importable (runs once)."""
    global _det3d_ready
    if _det3d_ready:
        return

    repo_dir = _find_patched_repo()
    if not repo_dir:
        repo_dir = _clone_and_patch()

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    _det3d_ready = True


_setup_det3d()

from det3d.models.necks.rpn import RPN  # noqa: E402
from det3d.models.bbox_heads.center_head import CenterHead  # noqa: E402


NUSCENES_TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]


HEAD_KEYS = ["reg", "height", "dim", "rot", "vel", "hm"]
TASKS_NUM_CLASSES = [1, 2, 2, 1, 2, 2]


class CenterPointOriginal(nn.Module):
    """Original det3d RPN + CenterHead for TT compilation.

    Uses the actual det3d classes from tianweiy/CenterPoint with
    CUDA ops monkey-patched out. Forward pass is pure PyTorch and
    fully XLA-traceable.

    Returns a single flat tensor (all task head outputs concatenated)
    so the test evaluator can do a reliable PCC comparison without
    pytree-matching issues across CPU and TT outputs.
    """

    def __init__(self):
        super().__init__()
        self.rpn = RPN(
            layer_nums=[3, 5, 5],
            ds_layer_strides=[2, 2, 2],
            ds_num_filters=[64, 128, 256],
            us_layer_strides=[0.5, 1, 2],
            us_num_filters=[128, 128, 128],
            num_input_features=64,
            logger=logging.getLogger("CenterPointOriginal.RPN"),
        )
        self.head = CenterHead(
            in_channels=sum([128, 128, 128]),
            tasks=NUSCENES_TASKS,
            dataset="nuscenes",
            weight=0.25,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            common_heads={
                "reg": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        rpn_out = self.rpn(bev)
        task_outputs, _ = self.head(rpn_out)
        flat = []
        for task_dict in task_outputs:
            for key in HEAD_KEYS:
                flat.append(task_dict[key].flatten())
        return torch.cat(flat)
