# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) Megvii Inc. All rights reserved.
import os
from argparse import ArgumentParser

import torch
import re
import ast
import copy

import sys as _sys
import os as _os

_pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)
from nusc_det_dataset import NuscDetDataset, collate_fn
from det_evaluators import DetNuscEvaluator
from base_bev_depth import BaseBEVDepth
from torch_dist import get_rank

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(
    img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True
)

backbone_conf = {
    "x_bound": [-51.2, 51.2, 0.8],
    "y_bound": [-51.2, 51.2, 0.8],
    "z_bound": [-5, 3, 8],
    "d_bound": [2.0, 58.0, 0.5],
    "final_dim": final_dim,
    "output_channels": 80,
    "downsample_factor": 16,
    "img_backbone_conf": dict(
        type="ResNet",
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    "img_neck_conf": dict(
        type="SECONDFPN",
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    "depth_net_conf": dict(in_channels=512, mid_channels=512),
}
ida_aug_conf = {
    "resize_lim": (0.386, 0.55),
    "final_dim": final_dim,
    "rot_lim": (-5.4, 5.4),
    "H": H,
    "W": W,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.0),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

bda_aug_conf = {
    "rot_lim": (-22.5, 22.5),
    "scale_lim": (0.95, 1.05),
    "flip_dx_ratio": 0.5,
    "flip_dy_ratio": 0.5,
}

bev_backbone = dict(
    type="ResNet",
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(
    type="SECONDFPN",
    in_channels=[80, 160, 320, 640],
    upsample_strides=[1, 2, 4, 8],
    out_channels=[64, 64, 64, 64],
)

CLASSES = [
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
]

TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))

bbox_coder = dict(
    type="CenterPointBBoxCoder",
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type="circle",
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    "bev_backbone_conf": bev_backbone,
    "bev_neck_conf": bev_neck,
    "tasks": TASKS,
    "common_heads": common_heads,
    "bbox_coder": bbox_coder,
    "train_cfg": train_cfg,
    "test_cfg": test_cfg,
    "in_channels": 256,  # Equal to bev_neck output_channels.
    "loss_cls": dict(type="GaussianFocalLoss", reduction="mean"),
    "loss_bbox": dict(type="L1Loss", reduction="mean", loss_weight=0.25),
    "gaussian_overlap": 0.1,
    "min_radius": 2,
}


def build_dataloader(
    data_root: str, batch_size: int, split: str, use_fusion: bool, key_idxes
):
    if split == "val":
        info_path = os.path.join(data_root, "nuscenes_infos_val.pkl")
    elif split == "test":
        info_path = os.path.join(data_root, "nuscenes_infos_test.pkl")
    else:
        raise ValueError("split must be one of {val, test}")

    dataset = NuscDetDataset(
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        data_root=data_root,
        info_paths=info_path,
        is_train=False,
        img_conf=img_conf,
        num_sweeps=1,
        sweep_idxes=list(),
        key_idxes=key_idxes or list(),
        return_depth=use_fusion,
        use_fusion=use_fusion,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: collate_fn(x, is_return_depth=use_fusion),
        sampler=None,
    )
    return loader


def load_checkpoint_if_provided(model: torch.nn.Module, ckpt_path: str):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.module."):
            new_key = k[len("model.module.") :]
        elif k.startswith("model."):
            new_key = k[len("model.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if get_rank() == 0:
        if missing:
            print(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading checkpoint: {unexpected}")


def parse_exp_overrides(exp_path: str):
    overrides = {
        "use_fusion": False,
        "key_idxes": [],
        "train_code_weights_len": None,
    }
    if not exp_path:
        return overrides
    # Accept module-like (bevdepth/exps/...py) or absolute path
    if os.path.isdir(exp_path):
        return overrides
    if not os.path.isfile(exp_path):
        return overrides
    try:
        with open(exp_path, "r") as f:
            content = f.read()
        # # Detect Fusion
        # if 'from bevdepth.models.fusion_bev_depth import FusionBEVDepth' in content:
        #     overrides['use_fusion'] = True
        # key_idxes assignments
        m = re.search(r"key_idxes\s*=\s*\[([^\]]*)\]", content)
        if m:
            list_str = "[" + m.group(1) + "]"
            try:
                overrides["key_idxes"] = ast.literal_eval(list_str)
            except Exception:
                overrides["key_idxes"] = []
        # code_weights length (e.g., 10 entries)
        m2 = re.search(r"code_weights\]\s*=\s*\[([^\]]+)\]", content)
        if m2:
            try:
                weights = ast.literal_eval("[" + m2.group(1) + "]")
                overrides["train_code_weights_len"] = len(weights)
            except Exception:
                pass
    except Exception:
        pass
    return overrides


def run_standalone():
    parser = ArgumentParser(description="BEVDepth standalone eval/predict")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/proj_sw/user_dev/mramanathan/bgdlab19_sep10_xla/tt-xla/tests/jax/single_chip/models/bevdepth_data/data/nuScenes",
    )
    parser.add_argument("-b", "--batch_size_per_device", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("-e", "--evaluate", action="store_true", help="run on val set")
    parser.add_argument("-p", "--predict", action="store_true", help="run on test set")
    parser.add_argument("--default_root_dir", type=str, default="./outputs/base_exp")
    parser.add_argument(
        "--exp",
        type=str,
        default="",
        help="Path to exp .py (e.g., bevdepth/exps/nuscenes/mv/xxx.py)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    os.makedirs(args.default_root_dir, exist_ok=True)
    evaluator = DetNuscEvaluator(class_names=CLASSES, output_dir=args.default_root_dir)

    exp_overrides = parse_exp_overrides(args.exp)

    # Clone configs so we can mutate safely per run
    backbone_conf_run = copy.deepcopy(backbone_conf)
    head_conf_run = copy.deepcopy(head_conf)

    # Apply key_idxes-driven channel changes (multi-key input)
    key_idxes = exp_overrides["key_idxes"]
    if key_idxes:
        multi_factor = len(key_idxes) + 1
        head_conf_run["bev_backbone_conf"]["in_channels"] = 80 * multi_factor
        head_conf_run["bev_neck_conf"]["in_channels"] = [
            80 * multi_factor,
            160,
            320,
            640,
        ]
        # If the exp typically uses 10 code weights, adjust if detected
        if exp_overrides["train_code_weights_len"]:
            head_conf_run["train_cfg"]["code_weights"] = [1.0] * exp_overrides[
                "train_code_weights_len"
            ]

    use_fusion = exp_overrides["use_fusion"]
    # if use_fusion:
    #     model = FusionBEVDepth(backbone_conf_run, head_conf_run, is_train_depth=False)
    # else:
    model = BaseBEVDepth(backbone_conf_run, head_conf_run, is_train_depth=False)
    model = model.to(device)
    model.eval()

    load_checkpoint_if_provided(model, args.ckpt_path)

    if args.predict:
        loader = build_dataloader(
            args.data_root, args.batch_size_per_device, "test", use_fusion, key_idxes
        )
    else:
        loader = build_dataloader(
            args.data_root, args.batch_size_per_device, "val", use_fusion, key_idxes
        )

    all_pred_results = []
    all_img_metas = []

    from tqdm import tqdm  # Add this at the top

    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Evaluating", leave=False
        ):  # Wrap loader with tqdm
            if use_fusion:
                sweep_imgs, mats, _, img_metas, _, _, lidar_depth = batch
            else:
                sweep_imgs, mats, _, img_metas, _, _ = batch

            for key, value in mats.items():
                mats[key] = value.to(device, non_blocking=True)
            sweep_imgs = sweep_imgs.to(device, non_blocking=True)
            if use_fusion:
                lidar_depth = lidar_depth.to(device, non_blocking=True)
                preds = model(sweep_imgs, mats, lidar_depth)
            else:
                preds = model(sweep_imgs, mats)
            print("preds = ", preds)
            exit()
    #         results = model.get_bboxes(preds, img_metas)

    #         for i in range(len(results)):
    #             results[i][0] = results[i][0].detach().cpu().numpy()
    #             results[i][1] = results[i][1].detach().cpu().numpy()
    #             results[i][2] = results[i][2].detach().cpu().numpy()
    #             all_pred_results.append(results[i][:3])
    #             all_img_metas.append(img_metas[i])

    # # Evaluate or format predictions
    # if args.predict:
    #     if not args.ckpt_path:
    #         output_dir = args.default_root_dir
    #     else:
    #         output_dir = os.path.dirname(args.ckpt_path)
    #     evaluator._format_bbox(all_pred_results, all_img_metas, output_dir)
    # else:
    #     if get_rank() == 0:
    #         evaluator.evaluate(all_pred_results, all_img_metas)


if __name__ == "__main__":
    run_standalone()
