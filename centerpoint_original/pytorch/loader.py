# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CenterPoint model loader — original det3d code from tianweiy/CenterPoint.

Uses the original det3d RPN and CenterHead classes with CUDA ops monkey-patched
out for CPU inference. Loads pretrained mmdetection3d nuScenes PointPillars weights.

Reference: https://github.com/tianweiy/CenterPoint (CVPR 2021)
"""

import torch
from typing import Optional
from dataclasses import dataclass

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
from ...tools.utils import get_file
from .src.model import CenterPointOriginal

CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/"
    "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
    "centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth"
)


@dataclass
class CenterPointOriginalConfig(ModelConfig):
    bev_channels: int = 64
    bev_h: int = 512
    bev_w: int = 512


class ModelVariant(StrEnum):
    CENTERPOINT_ORIGINAL = "CenterPoint_Original"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.CENTERPOINT_ORIGINAL: CenterPointOriginalConfig(
            pretrained_model_name="centerpoint_original",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CENTERPOINT_ORIGINAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CenterPoint_Original",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MULTIVIEW_3D_OBJECT_DET,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = CenterPointOriginal()
        self._load_weights(model)

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def _load_weights(self, model):
        """Load mmdetection3d checkpoint into the original det3d model.

        The mmdet3d checkpoint uses different key conventions than det3d:
          - pts_backbone → rpn.blocks (with +1 index shift for ZeroPad2d)
          - pts_neck → rpn.deblocks
          - pts_bbox_head → head (with task_heads→tasks, heatmap→hm,
            and conv/bn submodule flattening)
        """
        ckpt_path = get_file(CHECKPOINT_URL)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        rpn_sd = self._remap_rpn_keys(state_dict)
        head_sd = self._remap_head_keys(state_dict)

        model.rpn.load_state_dict(rpn_sd, strict=False)
        model.head.load_state_dict(head_sd, strict=False)

    @staticmethod
    def _remap_rpn_keys(state_dict: dict) -> dict:
        """Remap checkpoint keys for the det3d RPN.

        det3d RPN has ZeroPad2d at index 0 in each block, so all
        subsequent layer indices in the checkpoint are shifted by +1.
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

    @staticmethod
    def _remap_head_keys(state_dict: dict) -> dict:
        """Remap checkpoint keys for the det3d CenterHead.

        mmdet3d checkpoint layout differs from det3d in three ways:
          1. shared_conv uses named sub-modules (conv, bn) vs Sequential indices
          2. task_heads vs tasks naming
          3. heatmap vs hm naming
          4. SepHead layers use nested conv/bn sub-modules vs flat Sequential
        """
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

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate a realistic BEV pseudo-image input.

        Creates a BEV tensor (B, 64, 512, 512) with spatial statistics
        that mimic the output of the PointPillars scatter operation.
        Real nuScenes BEV data is mostly zero (unoccupied pillars) with
        non-zero features concentrated in a band around the ego vehicle.
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        bev = torch.zeros(batch_size, cfg.bev_channels, cfg.bev_h, cfg.bev_w, dtype=dtype)

        # Simulate occupied pillars in a ring around ego vehicle (center of BEV)
        cx, cy = cfg.bev_w // 2, cfg.bev_h // 2
        torch.manual_seed(42)
        num_pillars = 8000
        angles = torch.rand(num_pillars) * 2 * 3.14159
        radii = 30 + torch.rand(num_pillars) * 180
        px = (cx + radii * torch.cos(angles)).long().clamp(0, cfg.bev_w - 1)
        py = (cy + radii * torch.sin(angles)).long().clamp(0, cfg.bev_h - 1)
        features = torch.randn(num_pillars, cfg.bev_channels, dtype=dtype) * 0.5
        for i in range(num_pillars):
            bev[0, :, py[i], px[i]] = features[i]

        if batch_size > 1:
            bev = bev.expand(batch_size, -1, -1, -1).contiguous()

        return (bev,)
