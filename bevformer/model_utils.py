# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Any, Dict, List


def build_from_input_image(input_image: Dict[str, Any]) -> Dict[str, Any]:
    """Build the tensors and metadata dictionary expected by the BEV wrapper and test.

    The structure of input_image is produced by the loader's load_inputs method and
    follows the mmcv/mmdet data container pattern.
    """
    # Extract mmcv DataContainer-wrapped fields
    img_metas_data_container = input_image["img_metas"][0]
    img_metas = img_metas_data_container.data
    img_data_container = input_image["img"][0]

    # Pull metadata from the first sample
    meta0 = img_metas[0][0]
    filename = meta0["filename"]
    ori_shapes = meta0["ori_shape"]
    img_shapes = meta0["img_shape"]
    lidar2img = meta0["lidar2img"]
    lidar2cam = meta0["lidar2cam"]
    pad_shape = meta0["pad_shape"]
    box_mode_3d = meta0["box_mode_3d"]
    box_type_3d = meta0["box_type_3d"]
    img_norm_cfg = meta0["img_norm_cfg"]
    pts_filename = meta0["pts_filename"]
    can_bus = meta0["can_bus"]

    # Extract image tensor(s)
    img = img_data_container.data

    # Prepare tensors for wrapper forward signature
    ori_shapes_tensor = torch.tensor(ori_shapes, dtype=torch.float32).unsqueeze(0)
    img_shapes_tensor = torch.tensor(img_shapes, dtype=torch.float32).unsqueeze(0)
    pad_shapes_tensor = torch.tensor(pad_shape, dtype=torch.float32).unsqueeze(0)

    lidar2img_tensor_list: List[torch.Tensor] = [
        torch.tensor(arr, dtype=torch.float32) for arr in lidar2img
    ]
    lidar2img_stacked_tensor = torch.stack(lidar2img_tensor_list, dim=0).unsqueeze(0)

    lidar2cam_tensor_list: List[torch.Tensor] = [
        torch.tensor(arr, dtype=torch.float32) for arr in lidar2cam
    ]
    lidar2cam_stacked_tensor = torch.stack(lidar2cam_tensor_list, dim=0).unsqueeze(0)

    # Single image per batch; keep a batch dim for wrapper forward
    img_pybuda = img[0].unsqueeze(0)

    return {
        # Scalars/objects needed for wrapper constructor
        "filename": filename,
        "box_mode_3d": box_mode_3d,
        "box_type_3d": box_type_3d,
        "img_norm_cfg": img_norm_cfg,
        "pts_filename": pts_filename,
        "can_bus": can_bus,
        # Raw lists also passed to wrapper constructor by the test
        "lidar2img": lidar2img,
        "ori_shapes": ori_shapes,
        "lidar2cam": lidar2cam,
        "img_shapes": img_shapes,
        "pad_shape": pad_shape,
        # Tensors consumed by wrapper forward
        "ori_shapes_tensor": ori_shapes_tensor,
        "img_shapes_tensor": img_shapes_tensor,
        "lidar2img_stacked_tensor": lidar2img_stacked_tensor,
        "lidar2cam_stacked_tensor": lidar2cam_stacked_tensor,
        "pad_shapes_tensor": pad_shapes_tensor,
        "img_pybuda": img_pybuda,
        # Optional for debugging/reference
        "img_metas": img_metas,
        "img": img,
    }


class BEV_wrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        filename: Any,
        box_mode_3d: Any,
        box_type_3d: Any,
        img_norm_cfg: Dict[str, Any],
        pts_filename: Any,
        can_bus: Any,
        lidar2img: Any,
        ori_shapes: Any,
        lidar2cam: Any,
        img_shapes: Any,
        pad_shape: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.filename = filename
        self.box_mode_3d = box_mode_3d
        self.box_type_3d = box_type_3d
        self.img_norm_cfg = img_norm_cfg
        self.pts_filename = pts_filename
        self.can_bus = can_bus
        # Store additional fields for completeness/debugging
        self._lidar2img = lidar2img
        self._ori_shapes = ori_shapes
        self._lidar2cam = lidar2cam
        self._img_shapes = img_shapes
        self._pad_shape = pad_shape

    def forward(
        self,
        ori_shapes_tensor: torch.Tensor,
        img_shapes_tensor: torch.Tensor,
        lidar2img_stacked_tensor: torch.Tensor,
        lidar2cam_stacked_tensor: torch.Tensor,
        pad_shapes_tensor: torch.Tensor,
        img_pybuda: torch.Tensor,
    ):
        # Remove the extra batch dimension added during build
        lidar2img_stacked_tensor = lidar2img_stacked_tensor.squeeze(0)
        lidar2cam_stacked_tensor = lidar2cam_stacked_tensor.squeeze(0)
        ori_shapes_tensor = ori_shapes_tensor.squeeze(0)
        img_shapes_tensor = img_shapes_tensor.squeeze(0)
        pad_shapes_tensor = pad_shapes_tensor.squeeze(0)
        img_pybuda = img_pybuda.squeeze(0)

        # Convert tensors back to native Python structures for model metadata
        lidar2img_array = [tensor.numpy() for tensor in lidar2img_stacked_tensor]
        lidar2cam_array = [tensor.numpy() for tensor in lidar2cam_stacked_tensor]
        ori_shapes_list = [tuple(row.tolist()) for row in ori_shapes_tensor]
        img_shapes_list = [tuple(row.tolist()) for row in img_shapes_tensor]
        pad_shapes_list = [tuple(row.tolist()) for row in pad_shapes_tensor]

        img_metas = {
            "filename": self.filename,
            "ori_shape": ori_shapes_list,
            "img_shape": img_shapes_list,
            "lidar2img": lidar2img_array,
            "lidar2cam": lidar2cam_array,
            "pad_shape": pad_shapes_list,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            "box_mode_3d": self.box_mode_3d,
            "box_type_3d": self.box_type_3d,
            "img_norm_cfg": self.img_norm_cfg,
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            "pts_filename": self.pts_filename,
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            "can_bus": self.can_bus,
        }

        # Recreate the input dict structure expected by BEVFormer forward
        input_pybuda_dict = {
            "rescale": True,
            "img_metas": [[img_metas]],
            "img": [img_pybuda],
        }
        output = self.model(return_loss=False, **input_pybuda_dict)
        return output
