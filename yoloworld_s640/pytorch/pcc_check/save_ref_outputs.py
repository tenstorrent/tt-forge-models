# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step 1 of PCC check — save reference outputs.

Reference model:
  yoloworld/pytorch  (mmdet-free port of https://github.com/AILab-CVC/YOLO-World)
  Same checkpoint (Small_640.pth), same weights, same architecture as the original
  GitHub source. Used as ground truth for the PCC comparison.

Usage (run from tt-xla repo root):
    python third_party/tt_forge_models/yoloworld_s640/pytorch/pcc_check/save_ref_outputs.py
    python ...save_ref_outputs.py --out /tmp/ref_out.pt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

import torch


def main(out_path: str):
    import sys, os
    from third_party.tt_forge_models.tools.utils import get_file
    import third_party.tt_forge_models.yoloworld.pytorch.src.model  # registers all MODELS
    from third_party.tt_forge_models.yoloworld.pytorch.src.utils import (
        init_detector, Config, get_base_cfg,
    )

    print("Reference: yoloworld/pytorch  (AILab-CVC/YOLO-World port)")

    checkpoint = str(get_file("test_files/pytorch/yoloworld/Small_640.pth"))
    base_cfg = get_base_cfg(variant="Small_640")
    config = Config(base_cfg)
    config.load_from = checkpoint
    # palette="random" avoids the lvis dataset dependency (only needed for vis)
    model = init_detector(config, checkpoint, palette="random")
    model.eval()

    texts = [["person"], ["bus"], [" "]]
    model.reparameterize(texts)

    # Build image tensor the same way as yoloworld_s640/pytorch loader
    import cv2
    import numpy as np

    image_file = get_file("https://ultralytics.com/images/bus.jpg")
    img = cv2.imread(str(image_file))
    h, w = img.shape[:2]
    scale = min(640 / h, 640 / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_h = (640 - new_h) // 2
    pad_w = (640 - new_w) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized
    img_rgb = canvas[:, :, ::-1].copy()
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    print(f"  image tensor: {img_tensor.shape}  dtype: {img_tensor.dtype}")

    # The reference model._forward has a bug when data_samples=None (txt_masks unbound).
    # Replicate its logic: image backbone → neck → bbox_head, using cached text_feats.
    print("Running forward pass (manual: backbone.forward_image → neck → bbox_head)...")
    with torch.no_grad():
        img_feats = model.backbone.forward_image(img_tensor)
        txt_feats = model.text_feats  # set by reparameterize()
        neck_feats = model.neck(img_feats, txt_feats)
        cls_logits, bbox_preds = model.bbox_head.forward(neck_feats, txt_feats, None)

    print("Outputs:")
    for i, (cl, bp) in enumerate(zip(cls_logits, bbox_preds)):
        print(f"  level {i}:  cls {tuple(cl.shape)}  bbox {tuple(bp.shape)}")

    torch.save(
        {
            "cls_logits": [c.cpu() for c in cls_logits],
            "bbox_preds": [b.cpu() for b in bbox_preds],
            "image": img_tensor.cpu(),
            "source": "yoloworld/pytorch (AILab-CVC port)",
        },
        out_path,
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/yoloworld_ref_out.pt")
    args = ap.parse_args()
    main(args.out)
