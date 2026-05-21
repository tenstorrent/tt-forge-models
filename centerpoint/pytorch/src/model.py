# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterPoint model classes and utilities.

The patched det3d source lives at centerpoint/src/ (a copy of the upstream
CenterPoint repo with the cpu-inference patch already applied).  This module
adds that directory to sys.path and stubs out CUDA-only dependencies so that
RPN + CenterHead can be imported and executed on CPU or TT hardware.
"""

import importlib.machinery
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path to the patched CenterPoint source tree (centerpoint/src/)
_CENTERPOINT_SRC = Path(__file__).resolve().parent.parent.parent / "src"

# PointPillars / BEV constants (match mmdet3d checkpoint training config)
PC_RANGE: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
VOXEL_SIZE: Tuple[float, ...] = (0.2, 0.2, 8.0)
GRID_X: int = int((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0])  # 512
GRID_Y: int = int((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1])  # 512
MAX_POINTS_PER_PILLAR: int = 20
MAX_PILLARS: int = 30_000
NUM_INPUT_FEATURES: int = 5  # x, y, z, intensity, ring
OUT_SIZE_FACTOR: int = 4  # RPN downsamples 512 → 128

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/"
    "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
    "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth"
)

TASK_NAMES = ["car", "truck+vehicle", "bus+trailer", "barrier", "moto+bike", "ped+cone"]
TASK_COLORS = ["cyan", "orange", "magenta", "yellow", "lime", "red"]

_centerpoint_ready = False


# ---------------------------------------------------------------------------
# sys.path setup and CUDA stub
# ---------------------------------------------------------------------------


def _patch_spconv_stub():
    """Stub out spconv so det3d imports without a CUDA build.

    checkpoint.py in the CenterPoint source imports spconv at module level.
    CenterPoint (PointPillars variant) does not use sparse convolutions at
    runtime, so a lightweight stub is sufficient.
    """

    class _SparseModule(torch.nn.Module):
        pass

    class _SparseConvNd(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _SparseSequential(torch.nn.Sequential):
        pass

    class _SparseTensor:
        def __init__(self, *args, **kwargs):
            pass

    class _Ops:
        pass

    _SPCONV_ATTRS = {
        "ops": _Ops(),
        "SparseModule": _SparseModule,
        "SparseConv3d": _SparseConvNd,
        "SubMConv3d": _SparseConvNd,
        "SparseSequential": _SparseSequential,
        "SparseTensor": _SparseTensor,
    }
    for name in ("spconv", "spconv.pytorch"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            for attr, val in _SPCONV_ATTRS.items():
                setattr(stub, attr, val)
            sys.modules[name] = stub


def _init_centerpoint():
    """Add the patched CenterPoint source to sys.path (runs once)."""
    global _centerpoint_ready
    if _centerpoint_ready:
        return

    src_dir = str(_CENTERPOINT_SRC)
    if not _CENTERPOINT_SRC.is_dir():
        raise FileNotFoundError(
            f"CenterPoint source not found at {src_dir}. "
            "Expected centerpoint/src/ to contain the patched det3d tree."
        )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _patch_spconv_stub()
    _centerpoint_ready = True


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _download_weights() -> Path:
    """Download the mmdetection3d checkpoint and return its local path."""
    import os

    cache_dir = (
        Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))
        / "centerpoint"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = CHECKPOINT_URL.rsplit("/", 1)[-1]
    local_path = cache_dir / filename
    if not local_path.exists():
        print(f"Downloading CenterPoint weights to {local_path} ...")
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(local_path))
    return local_path


def _load_checkpoint() -> dict:
    """Download and return the mmdetection3d state dict."""
    ckpt_path = _download_weights()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get("state_dict", ckpt)


# ---------------------------------------------------------------------------
# Weight remapping: mmdet3d → det3d naming conventions
# ---------------------------------------------------------------------------


def _remap_pfn_keys(state_dict: dict) -> dict:
    return {
        k.replace("pts_voxel_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("pts_voxel_encoder.")
    }


def _remap_rpn_keys(state_dict: dict) -> dict:
    """Remap mmdet3d backbone/neck keys for the det3d RPN.

    det3d inserts a ZeroPad2d at index 0 in each block, so checkpoint layer
    indices are shifted by +1.
    """
    rpn_sd = {}
    for k, v in state_dict.items():
        if k.startswith("pts_backbone."):
            new_k = k.replace("pts_backbone.", "")
            parts = new_k.split(".")
            if len(parts) >= 3 and parts[0] == "blocks":
                try:
                    parts[2] = str(int(parts[2]) + 1)
                    new_k = ".".join(parts)
                except ValueError:
                    pass
            rpn_sd[new_k] = v
        elif k.startswith("pts_neck."):
            rpn_sd[k.replace("pts_neck.", "")] = v
    return rpn_sd


def _remap_head_keys(state_dict: dict) -> dict:
    """Remap mmdet3d bbox-head keys for the det3d CenterHead."""
    head_sd = {}
    for k, v in state_dict.items():
        if not k.startswith("pts_bbox_head."):
            continue
        new_k = k.replace("pts_bbox_head.", "")
        new_k = new_k.replace("shared_conv.conv.", "shared_conv.0.")
        new_k = new_k.replace("shared_conv.bn.", "shared_conv.1.")
        new_k = new_k.replace("task_heads.", "tasks.")
        new_k = new_k.replace(".heatmap.", ".hm.")

        parts = new_k.split(".")
        if len(parts) >= 5 and parts[0] == "tasks":
            t, name = parts[1], parts[2]
            layer_idx = int(parts[3])
            rest = parts[4:]
            if rest[0] == "conv":
                new_k = f"tasks.{t}.{name}.{layer_idx * 3}.{'.'.join(rest[1:])}"
            elif rest[0] == "bn":
                new_k = f"tasks.{t}.{name}.{layer_idx * 3 + 1}.{'.'.join(rest[1:])}"
            else:
                new_k = f"tasks.{t}.{name}.{layer_idx * 3}.{'.'.join(rest)}"

        head_sd[new_k] = v
    return head_sd


# ---------------------------------------------------------------------------
# CenterPointRPNHead: RPN + CenterHead (primary compilation target)
# ---------------------------------------------------------------------------

_TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

_COMMON_HEADS = {
    "reg": (2, 2),
    "height": (1, 2),
    "dim": (3, 2),
    "rot": (2, 2),
    "vel": (2, 2),
}


class CenterPointRPNHead(nn.Module):
    """RPN backbone + CenterHead detection head.

    Input:  BEV feature map  (B, 64, 512, 512)
    Output: List[Dict[str, Tensor]] — one dict per task, keys are
            'hm', 'reg', 'height', 'dim', 'rot', 'vel'
    """

    def __init__(self):
        super().__init__()
        import logging

        from det3d.models.necks.rpn import RPN
        from det3d.models.bbox_heads.center_head import CenterHead

        self.rpn = RPN(
            layer_nums=[3, 5, 5],
            ds_layer_strides=[2, 2, 2],
            ds_num_filters=[64, 128, 256],
            us_layer_strides=[0.5, 1, 2],
            us_num_filters=[128, 128, 128],
            num_input_features=64,
            logger=logging.getLogger("RPN"),
        )
        self.head = CenterHead(
            in_channels=sum([128, 128, 128]),
            tasks=_TASKS,
            dataset="nuscenes",
            weight=0.25,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            common_heads=_COMMON_HEADS,
        )

    def forward(self, bev: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        task_outputs, _ = self.head(self.rpn(bev))
        return task_outputs


def load_model_with_weights(dtype: Optional[torch.dtype] = torch.bfloat16) -> CenterPointRPNHead:
    """Create CenterPointRPNHead with pretrained mmdetection3d weights."""
    _init_centerpoint()
    model = CenterPointRPNHead()
    sd = _load_checkpoint()
    model.rpn.load_state_dict(_remap_rpn_keys(sd), strict=False)
    model.head.load_state_dict(_remap_head_keys(sd), strict=False)
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    return model


def get_single_input(
    dtype: torch.dtype = torch.bfloat16, batch_size: int = 1
) -> torch.Tensor:
    """Return a synthetic BEV feature map (B, 64, 512, 512)."""
    return torch.randn(batch_size, 64, 512, 512, dtype=dtype)


# ---------------------------------------------------------------------------
# PillarFeatureNet + PointPillarsScatter (pure PyTorch, no CUDA)
# ---------------------------------------------------------------------------


class _PFNLayerCPU(nn.Module):
    """Single PFN layer: Linear → BN1d → ReLU → channel-wise max-pool."""

    def __init__(self, in_ch: int, out_ch: int, last_layer: bool = True):
        super().__init__()
        self.last_vfe = last_layer
        units = out_ch if last_layer else out_ch // 2
        self.units = units
        self.linear = nn.Linear(in_ch, units, bias=False)
        self.norm = nn.BatchNorm1d(units, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        N, P, _ = x.shape
        out = self.linear(x.view(N * P, -1)).view(N, P, self.units)
        out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = F.relu(out)
        out = out * mask.unsqueeze(-1)
        x_max = out.max(dim=1)[0]
        if self.last_vfe:
            return x_max
        x_rep = x_max.unsqueeze(1).expand_as(out)
        return torch.cat([out, x_rep], dim=-1)


class PillarFeatureNetCPU(nn.Module):
    """PointPillars PillarFeatureNet (pure PyTorch, no CUDA ops).

    Matches the mmdet3d checkpoint: linear(10, 64) + BN1d(64).
    Input features per point: [x, y, z, intensity, ring,
                                x-cx, y-cy, z-cz,     ← cluster offsets
                                cx_off, cy_off]        ← pillar-centre offsets
    """

    def __init__(
        self,
        num_input: int = NUM_INPUT_FEATURES,
        num_filters: Tuple[int, ...] = (64,),
    ):
        super().__init__()
        aug_in = num_input + 5  # +3 cluster offset, +2 pillar-centre offset
        sizes = [aug_in] + list(num_filters)
        layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            layers.append(_PFNLayerCPU(sizes[i], sizes[i + 1], last_layer=last))
        self.pfn_layers = nn.ModuleList(layers)
        self.vx = VOXEL_SIZE[0]
        self.vy = VOXEL_SIZE[1]
        self.x_offset = self.vx / 2 + PC_RANGE[0]
        self.y_offset = self.vy / 2 + PC_RANGE[1]

    def forward(
        self,
        features: torch.Tensor,   # (N, max_pts, num_input)
        num_points: torch.Tensor,  # (N,)
        coors: torch.Tensor,       # (N, 4): [batch, 0, y_idx, x_idx]
    ) -> torch.Tensor:             # (N, 64)
        N, P, _ = features.shape
        dtype = features.dtype

        pts_mean = features[:, :, :3].sum(1, keepdim=True) / num_points.float().view(N, 1, 1)
        f_cluster = features[:, :, :3] - pts_mean

        f_center = torch.zeros(N, P, 2, dtype=dtype, device=features.device)
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset
        )

        feat = torch.cat([features, f_cluster, f_center], dim=-1)
        idx = torch.arange(P, device=features.device).unsqueeze(0).expand(N, -1)
        mask = (idx < num_points.unsqueeze(1)).float()
        feat = feat * mask.unsqueeze(-1)

        for pfn in self.pfn_layers:
            feat = pfn(feat, mask)
        return feat


class _PointPillarsScatterCPU(nn.Module):
    """Scatter pillar features onto a dense BEV canvas."""

    def forward(
        self,
        features: torch.Tensor,  # (N, C)
        coors: torch.Tensor,     # (N, 4): [batch, 0, y_idx, x_idx]
        batch_size: int,
        ny: int,
        nx: int,
    ) -> torch.Tensor:           # (B, C, ny, nx)
        C = features.shape[1]
        canvas = features.new_zeros(batch_size, C, ny * nx)
        for b in range(batch_size):
            m = coors[:, 0] == b
            c = coors[m]
            idx = (c[:, 2] * nx + c[:, 3]).long()
            canvas[b, :, idx] = features[m].t()
        return canvas.view(batch_size, C, ny, nx)


# ---------------------------------------------------------------------------
# Full pipeline: PFN → Scatter → RPN → CenterHead
# ---------------------------------------------------------------------------


class CenterPointFull(nn.Module):
    """Complete CenterPoint pipeline.

    forward(voxels, coords, num_pts, batch_size) → (bev, task_outputs)
    """

    def __init__(self):
        super().__init__()
        import logging

        from det3d.models.necks.rpn import RPN
        from det3d.models.bbox_heads.center_head import CenterHead

        self.pfn = PillarFeatureNetCPU()
        self.scatter = _PointPillarsScatterCPU()
        self.rpn = RPN(
            layer_nums=[3, 5, 5],
            ds_layer_strides=[2, 2, 2],
            ds_num_filters=[64, 128, 256],
            us_layer_strides=[0.5, 1, 2],
            us_num_filters=[128, 128, 128],
            num_input_features=64,
            logger=logging.getLogger("RPN"),
        )
        self.head = CenterHead(
            in_channels=sum([128, 128, 128]),
            tasks=_TASKS,
            dataset="nuscenes",
            weight=0.25,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            common_heads=_COMMON_HEADS,
        )

    def forward(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        num_pts: torch.Tensor,
        batch_size: int = 1,
    ):
        feat = self.pfn(voxels, num_pts, coords)
        bev = self.scatter(feat, coords, batch_size, GRID_Y, GRID_X)
        rpn_out = self.rpn(bev)
        task_outputs, _ = self.head(rpn_out)
        return bev, task_outputs


def load_full_model(dtype: Optional[torch.dtype] = torch.float32) -> CenterPointFull:
    """Create the complete CenterPoint pipeline with pretrained weights."""
    _init_centerpoint()
    model = CenterPointFull()
    sd = _load_checkpoint()
    model.pfn.load_state_dict(_remap_pfn_keys(sd), strict=False)
    model.rpn.load_state_dict(_remap_rpn_keys(sd), strict=False)
    model.head.load_state_dict(_remap_head_keys(sd), strict=False)
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    return model


# ---------------------------------------------------------------------------
# Voxelization (point cloud → pillars, pure numpy)
# ---------------------------------------------------------------------------


def voxelize(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a raw point cloud (N, 5) to PointPillars pillar tensors.

    Returns:
        voxels:     (V, MAX_POINTS_PER_PILLAR, 5) float32
        coords:     (V, 4) int32   — [0, 0, y_idx, x_idx]
        num_points: (V,) int32
    """
    xmin, ymin, zmin, xmax, ymax, zmax = PC_RANGE
    vx, vy, _ = VOXEL_SIZE

    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] < xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] < ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] < zmax)
    )
    pts = points[mask]

    ix = np.clip(np.floor((pts[:, 0] - xmin) / vx).astype(np.int32), 0, GRID_X - 1)
    iy = np.clip(np.floor((pts[:, 1] - ymin) / vy).astype(np.int32), 0, GRID_Y - 1)

    pillar_map: Dict[Tuple[int, int], List[int]] = {}
    for i, (px, py) in enumerate(zip(ix, iy)):
        key = (int(py), int(px))
        if key not in pillar_map:
            if len(pillar_map) >= MAX_PILLARS:
                continue
            pillar_map[key] = []
        if len(pillar_map[key]) < MAX_POINTS_PER_PILLAR:
            pillar_map[key].append(i)

    V = len(pillar_map)
    voxels = np.zeros((V, MAX_POINTS_PER_PILLAR, NUM_INPUT_FEATURES), dtype=np.float32)
    coords = np.zeros((V, 4), dtype=np.int32)
    num_pts = np.zeros(V, dtype=np.int32)

    for vi, ((py, px), indices) in enumerate(pillar_map.items()):
        n = len(indices)
        voxels[vi, :n] = pts[indices]
        coords[vi] = [0, 0, py, px]
        num_pts[vi] = n

    return voxels, coords, num_pts


# ---------------------------------------------------------------------------
# nuScenes data loading helper
# ---------------------------------------------------------------------------


def load_nuscenes_sample(
    nusc,
    sample_idx: int = 0,
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a nuScenes sample and return voxelized pillar tensors.

    Args:
        nusc:       NuScenes instance (v1.0-mini)
        sample_idx: which sample to load

    Returns:
        points:    raw (N, 5) float32 array (LiDAR sensor frame)
        voxels_t:  (V, MAX_POINTS_PER_PILLAR, 5) float32 tensor
        coords_t:  (V, 4) int32 tensor
        num_pts_t: (V,) int32 tensor
    """
    sample = nusc.sample[sample_idx]
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = nusc.get_sample_data_path(lidar_token)
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    voxels_np, coords_np, num_pts_np = voxelize(points)
    return (
        points,
        torch.from_numpy(voxels_np),
        torch.from_numpy(coords_np),
        torch.from_numpy(num_pts_np),
    )


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def postprocess(
    task_outputs: List[Dict[str, torch.Tensor]],
    score_threshold: float = 0.1,
    voxel_size: Tuple[float, float] = (0.2, 0.2),
    pc_range: Tuple[float, float] = (-51.2, -51.2),
    out_size_factor: int = 4,
) -> List[Dict[str, torch.Tensor]]:
    """Decode CenterPoint heatmaps to 3D box predictions."""
    results = []
    dx, dy = voxel_size
    x_min, y_min = pc_range

    for task in task_outputs:
        hm = torch.sigmoid(task["hm"].float())
        reg = task["reg"].float()
        height = task["height"].float()
        dim = task["dim"].float()
        rot = task["rot"].float()
        vel = task["vel"].float()

        B, C, H, W = hm.shape
        scores, labels = hm.max(dim=1)
        scores_flat = scores.view(B, -1)
        labels_flat = labels.view(B, -1)

        for b in range(B):
            mask = scores_flat[b] > score_threshold
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                results.append(
                    {
                        "boxes": torch.zeros(0, 9),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long),
                    }
                )
                continue

            ys = (idx // W).float()
            xs = (idx % W).float()
            reg_flat = reg[b].view(2, -1)[:, idx]
            cx = (xs + reg_flat[0]) * out_size_factor * dx + x_min
            cy = (ys + reg_flat[1]) * out_size_factor * dy + y_min
            cz = height[b].view(1, -1)[:, idx][0]
            dim_flat = torch.exp(dim[b].view(3, -1)[:, idx])
            rot_flat = rot[b].view(2, -1)[:, idx]
            vel_flat = vel[b].view(2, -1)[:, idx]
            yaw = torch.atan2(rot_flat[0], rot_flat[1])
            boxes = torch.stack(
                [cx, cy, cz, dim_flat[0], dim_flat[1], dim_flat[2], yaw,
                 vel_flat[0], vel_flat[1]],
                dim=1,
            )
            results.append(
                {
                    "boxes": boxes,
                    "scores": scores_flat[b][idx],
                    "labels": labels_flat[b][idx],
                }
            )

    return results


# ---------------------------------------------------------------------------
# BEV visualization
# ---------------------------------------------------------------------------


def _draw_rotated_box(ax, cx, cy, w, l, yaw, color="lime", alpha=0.8):
    """Draw a single rotated 2-D bounding box on a matplotlib axis."""
    import matplotlib.pyplot as plt

    cos_a, sin_a = math.cos(yaw), math.sin(yaw)
    corners_local = np.array(
        [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]
    )
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = (rot @ corners_local.T).T + np.array([cx, cy])
    poly = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=0.9, alpha=alpha)
    ax.add_patch(poly)


def plot_bev_detections(
    task_outputs: List[Dict[str, torch.Tensor]],
    detections: List[Dict[str, torch.Tensor]],
    save_path: str = "centerpoint_bev_detections.png",
    points: Optional[np.ndarray] = None,
    title: str = "CenterPoint — BEV Detections",
) -> str:
    """Render detected 3D boxes in bird's-eye view and save to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")

    if points is not None:
        xmin, ymin, _, xmax, ymax, _ = PC_RANGE
        in_range = (
            (points[:, 0] >= xmin)
            & (points[:, 0] < xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] < ymax)
        )
        pts = points[in_range]
        z = pts[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-6)
        ax.scatter(pts[:, 0], pts[:, 1], c=z_norm, cmap="gray", s=0.15, alpha=0.35, linewidths=0)

    for task_i, task in enumerate(task_outputs):
        hm = torch.sigmoid(task["hm"].float())
        hm_max = hm[0].max(0)[0].detach().numpy()
        ax.imshow(
            hm_max,
            cmap="inferno",
            origin="lower",
            extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
            vmin=0.05,
            vmax=hm_max.max().clip(min=0.1),
            alpha=0.12,
            interpolation="bilinear",
        )

    total_drawn = 0
    for task_i, det in enumerate(detections):
        boxes = det["boxes"]
        scores = det["scores"]
        color = TASK_COLORS[task_i]
        n_drawn = 0
        for j in range(len(boxes)):
            cx = boxes[j, 0].item()
            cy = boxes[j, 1].item()
            w = boxes[j, 3].item()
            l = boxes[j, 4].item()
            yaw = boxes[j, 6].item()
            score = scores[j].item()
            if w <= 0 or l <= 0 or w > 20 or l > 20:
                continue
            _draw_rotated_box(ax, cx, cy, w, l, yaw, color=color, alpha=min(score + 0.25, 1.0))
            n_drawn += 1
        total_drawn += n_drawn
        if n_drawn > 0:
            ax.plot([], [], color=color, linewidth=1.2, label=f"{TASK_NAMES[task_i]} ({n_drawn})")

    ax.plot(0, 0, "w+", markersize=12, markeredgewidth=2, zorder=10)
    ax.set_xlim(PC_RANGE[0], PC_RANGE[3])
    ax.set_ylim(PC_RANGE[1], PC_RANGE[4])
    ax.set_aspect("equal")
    ax.set_xlabel("X  (m)", color="white")
    ax.set_ylabel("Y  (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(f"{title}  [total={total_drawn}]", color="white", pad=8)

    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.35)
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    out = str(Path(save_path).resolve())
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close()
    return out
