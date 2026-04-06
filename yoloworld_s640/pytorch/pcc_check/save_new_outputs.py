# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step 2 of PCC check: run the new YOLO-World S640 (yoloworld_s640/pytorch — clean
rewrite) and save its raw forward outputs to a .pt file.

Usage:
    python save_new_outputs.py [--out /tmp/new_out.pt] [--weight-report]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

import torch


def main(out_path: str, weight_report: bool):
    from third_party.tt_forge_models.yoloworld_s640.pytorch.loader import ModelLoader

    loader = ModelLoader()
    print(f"Loading new model (yoloworld_s640/pytorch) — variant: {loader.DEFAULT_VARIANT}")

    # Load model with verbose weight-loading info
    from third_party.tt_forge_models.tools.utils import get_file
    from third_party.tt_forge_models.yoloworld_s640.pytorch.src.model import build_yoloworld_s640

    checkpoint = str(get_file("test_files/pytorch/yoloworld/Small_640.pth"))
    print(f"  checkpoint: {checkpoint}")

    model = build_yoloworld_s640()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)

    if weight_report:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"\nMissing keys ({len(result.missing_keys)}):")
            for k in result.missing_keys[:20]:
                print(f"  {k}")
            if len(result.missing_keys) > 20:
                print(f"  ... and {len(result.missing_keys) - 20} more")
        if result.unexpected_keys:
            print(f"\nUnexpected keys ({len(result.unexpected_keys)}):")
            for k in result.unexpected_keys[:20]:
                print(f"  {k}")
            if len(result.unexpected_keys) > 20:
                print(f"  ... and {len(result.unexpected_keys) - 20} more")
    else:
        model.load_state_dict(state_dict, strict=False)

    texts = [["person"], ["bus"], [" "]]
    model.to(torch.float32)
    model.reparameterize(texts)
    model.eval()

    inputs = loader.load_inputs()  # [1, 3, 640, 640]
    print(f"  image tensor: {inputs.shape}  dtype: {inputs.dtype}")

    print("Running forward pass...")
    with torch.no_grad():
        cls_logits, bbox_preds = model(inputs)

    print("New model outputs:")
    for i, (cl, bp) in enumerate(zip(cls_logits, bbox_preds)):
        print(f"  level {i}:  cls {tuple(cl.shape)}  bbox {tuple(bp.shape)}")

    payload = {
        "cls_logits": [c.cpu() for c in cls_logits],
        "bbox_preds": [b.cpu() for b in bbox_preds],
        "image": inputs.cpu(),
    }
    torch.save(payload, out_path)
    print(f"\nSaved new model outputs → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/yoloworld_new_out.pt")
    ap.add_argument("--weight-report", action="store_true",
                    help="Print missing/unexpected weight keys")
    args = ap.parse_args()
    main(args.out, args.weight_report)
