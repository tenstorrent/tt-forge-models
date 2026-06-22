# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone per-component on-device check for LongCat-Image.

The dynamic runner (tests/runner/test_models.py) computes its PCC golden by
running the model on the host CPU in the model's native bf16. For the 6 B
LongCat transformer / 7.7 B Qwen2.5-VL text encoder / 1024-res VAE decoder that
golden takes 7-40 min in eager bf16 and blows the per-test device-hang timeout
before device compilation even starts. This wrapper instead computes the golden
once in fp32 (fast where it fits) and runs the component on the TT device in
bf16, so the device path can be validated independently of the slow golden.

Usage:
    python3 test_device.py <text_encoder|transformer|vae> [--golden fp32|bf16|none]
"""

import argparse
import sys
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, __file__.rsplit("/tt_forge_models/", 1)[0])

from tt_forge_models.longcat_image.pytorch.loader import ModelLoader, ModelVariant

_VARIANTS = {
    "text_encoder": ModelVariant.TEXT_ENCODER,
    "transformer": ModelVariant.TRANSFORMER,
    "vae": ModelVariant.VAE,
}


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("component", choices=list(_VARIANTS))
    ap.add_argument("--golden", choices=["fp32", "bf16", "none"], default="fp32")
    args = ap.parse_args()
    variant = _VARIANTS[args.component]

    loader = ModelLoader(variant)

    # ---- golden (host) ----
    golden = None
    if args.golden != "none":
        gdt = torch.float32 if args.golden == "fp32" else torch.bfloat16
        t = time.time()
        gm = loader.load_model(dtype_override=gdt).eval()
        gin = loader.load_inputs(dtype_override=gdt)
        with torch.no_grad():
            go = gm(*gin)
        golden = (go if isinstance(go, torch.Tensor) else go[0]).float()
        print(f"[golden {args.golden}] {tuple(golden.shape)} in {round(time.time()-t)}s "
              f"finite={torch.isfinite(golden).all().item()}", flush=True)
        del gm, gin
        import gc

        gc.collect()

    # ---- device (bf16) ----
    xr.runtime.set_device_type("TT")
    device = torch_xla.device(0)
    print(f"[device] {device} chips={len(xm.get_xla_supported_devices())}", flush=True)

    t = time.time()
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)
    inputs = [x.to(device) for x in loader.load_inputs(dtype_override=torch.bfloat16)]
    print(f"[device] moved weights+inputs in {round(time.time()-t)}s", flush=True)

    t = time.time()
    with torch.no_grad():
        out = model(*inputs)
    out = out if isinstance(out, torch.Tensor) else out[0]
    torch_xla.sync()
    out_cpu = out.cpu().float()
    print(f"[device] forward+compile in {round(time.time()-t)}s shape={tuple(out_cpu.shape)} "
          f"finite={torch.isfinite(out_cpu).all().item()}", flush=True)

    if golden is not None:
        if golden.shape == out_cpu.shape:
            print(f"[PCC] {pcc(golden, out_cpu):.5f}", flush=True)
        else:
            print(f"[PCC] shape mismatch golden={tuple(golden.shape)} dev={tuple(out_cpu.shape)}", flush=True)
    print(f"{args.component.upper()} DEVICE OK", flush=True)


if __name__ == "__main__":
    main()
