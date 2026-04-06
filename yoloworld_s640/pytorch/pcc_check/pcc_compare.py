# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step 3 of PCC check — compare reference vs new model outputs.

Usage:
    python pcc_compare.py [--ref /tmp/yoloworld_ref_out.pt] [--new /tmp/yoloworld_new_out.pt]
"""

import argparse
import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a_mean = a.mean()
    b_mean = b.mean()
    num = ((a - a_mean) * (b - b_mean)).sum()
    den = (((a - a_mean) ** 2).sum() * ((b - b_mean) ** 2).sum()).sqrt()
    if den == 0:
        return float("nan")
    return (num / den).item()


def main(ref_path: str, new_path: str):
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)

    ref_cls = ref["cls_logits"]
    ref_bbox = ref["bbox_preds"]
    new_cls = new["cls_logits"]
    new_bbox = new["bbox_preds"]

    print(f"Reference: {ref.get('source', ref_path)}")
    print(f"New model: {new_path}")
    print()

    # Check image alignment
    if "image" in ref and "image" in new:
        img_pcc = pcc(ref["image"], new["image"])
        print(f"Input image PCC: {img_pcc:.6f}  (should be 1.0 — same preprocessing)")
        if img_pcc < 0.9999:
            print("  WARNING: input images differ — PCC comparison may be invalid")
    print()

    all_pass = True
    threshold = 0.99

    print(f"{'Level':<8} {'Tensor':<12} {'Ref shape':<20} {'New shape':<20} {'PCC':>10}  {'Pass?'}")
    print("-" * 80)

    for i, (rc, nc) in enumerate(zip(ref_cls, new_cls)):
        p = pcc(rc, nc)
        passed = p >= threshold
        all_pass = all_pass and passed
        print(f"  {i:<6} {'cls_logits':<12} {str(tuple(rc.shape)):<20} {str(tuple(nc.shape)):<20} {p:>10.6f}  {'PASS' if passed else 'FAIL'}")

    for i, (rb, nb) in enumerate(zip(ref_bbox, new_bbox)):
        p = pcc(rb, nb)
        passed = p >= threshold
        all_pass = all_pass and passed
        print(f"  {i:<6} {'bbox_preds':<12} {str(tuple(rb.shape)):<20} {str(tuple(nb.shape)):<20} {p:>10.6f}  {'PASS' if passed else 'FAIL'}")

    print()
    print(f"Overall: {'ALL PASS (PCC >= {})'.format(threshold) if all_pass else 'FAIL — some outputs below threshold'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/yoloworld_ref_out.pt")
    ap.add_argument("--new", default="/tmp/yoloworld_new_out.pt")
    args = ap.parse_args()
    raise SystemExit(main(args.ref, args.new))
