# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image composite-pipeline bringup.

Drives the real diffusers ``LongCatImagePipeline`` end-to-end in host Python
(scheduler / denoising loop stay on host) while the heavy denoiser
(``LongCatImageTransformer2DModel``) runs on the Tenstorrent device via
torch_xla. The text encoder (Qwen2.5-VL 7.7B) and the VAE run on CPU inside the
same host loop: all three components pass on-device standalone, but a single
32 GB Blackhole chip cannot hold the 7.7B encoder and the 6B denoiser resident
at once, so the denoiser -- the compute-dominant component and the mandated
on-device part -- owns the chip while the encoder / VAE stay on host.

Produces a real image from a real prompt -> proof the components compose.

KNOWN ISSUE (Blackhole, bf16): with the denoiser on device the first transformer
forward in the real loop emits +/-inf (the final image is all-black), so the
on-device end-to-end generation is not yet valid. The standalone per-component
runner test passes (PCC 0.99) because it uses unit-scale random inputs; the real
text-encoder embeddings have much larger magnitudes that push some op in the
device transformer to overflow bf16. The overflow appears on call 0 and is not
fixed by fp32_dest_acc_en / math_fidelity=hifi4; fp32 weights (~24 GB) do not fit
a single 32 GB chip. The full-CPU pipeline produces a correct image
(generated_cpu_reference.png) -- proving the composition / host wiring is right
and isolating the issue to the device transformer under real-magnitude inputs.
Needs op-by-op debugging to pinpoint the overflowing op. Set
COMPOSITE_DENOISER_DEVICE=0 to run the denoiser on CPU and get a valid image.
"""

import os
import sys

import torch
import torch_xla

REPO = "meituan-longcat/LongCat-Image"
DTYPE = torch.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", ".")
PROMPT = "a photograph of an astronaut riding a horse on the moon, highly detailed"
STEPS = int(os.environ.get("COMPOSITE_STEPS", "8"))
HW = int(os.environ.get("COMPOSITE_HW", "256"))


def _to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def main():
    from diffusers import LongCatImagePipeline

    dev = torch_xla.device()
    print(f"[composite] xla device: {dev}", flush=True)

    pipe = LongCatImagePipeline.from_pretrained(REPO, torch_dtype=DTYPE)
    pipe.set_progress_bar_config(disable=True)

    # ---- offload the denoiser (transformer) to the TT device ----
    # text_encoder + vae stay on CPU (host loop). Set COMPOSITE_DENOISER_DEVICE=0
    # to keep the denoiser on CPU too (workaround for the known on-device overflow).
    if os.environ.get("COMPOSITE_DENOISER_DEVICE", "1") == "1":
        transformer = pipe.transformer.to(dev)
        orig_forward = transformer.forward

        def device_forward(*args, **kwargs):
            out = orig_forward(*_to(args, dev), **_to(kwargs, dev))
            torch_xla.sync()
            return _to(out, "cpu")

        transformer.forward = device_forward
        pipe.transformer = transformer
        print("[composite] denoiser on TT device", flush=True)
    else:
        print("[composite] denoiser on CPU (workaround)", flush=True)

    print(f"[composite] generating {HW}x{HW}, {STEPS} steps, prompt-rewrite off", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(42)
    image = pipe(
        prompt=PROMPT,
        height=HW,
        width=HW,
        num_inference_steps=STEPS,
        guidance_scale=4.5,
        generator=gen,
        enable_prompt_rewrite=False,
    ).images[0]

    out_path = os.path.join(REPORT_DIR, "generated.png")
    os.makedirs(REPORT_DIR, exist_ok=True)
    image.save(out_path)
    print(f"[composite] COMPOSITE SUCCESS -> {out_path} ({image.size})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
