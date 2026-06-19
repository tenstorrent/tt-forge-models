# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU reference generation for LongCat-Image at native 1024x1024.

Used to produce a visual artifact when the TT device is unavailable. Runs the
exact same host-Python pipeline loop as the device composite; the three
components (validated independently on device) here run on CPU. Native default
resolution 1024x1024, prompt-rewrite disabled.
"""
import os
import sys
import time

import torch

REPO_ID = "meituan-longcat/LongCat-Image"
OUT = sys.argv[1] if len(sys.argv) > 1 else "generated.png"
STEPS = int(os.environ.get("LONGCAT_STEPS", "12"))
torch.set_grad_enabled(False)

# Disable the pipeline's torch_xla mark_step hook: this is a pure-CPU render and
# the TT board is wedged, so any device probe (mark_step -> populateDevices) would
# fault on the dead PCIe link.
import diffusers.pipelines.longcat_image.pipeline_longcat_image as _P

_P.XLA_AVAILABLE = False

from diffusers import LongCatImagePipeline

t0 = time.time()
pipe = LongCatImagePipeline.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16)
print(f"loaded {time.time()-t0:.0f}s", flush=True)
image = pipe(
    prompt=(
        "A photorealistic close-up of a fluffy orange cat wearing tiny round "
        "glasses, sitting at a wooden desk and reading a book, warm window light."
    ),
    height=1024,
    width=1024,
    num_inference_steps=STEPS,
    guidance_scale=4.5,
    enable_prompt_rewrite=False,
    output_type="pil",
).images[0]
image.save(OUT)
print(f"SAVED {OUT} size={image.size} ({time.time()-t0:.0f}s)", flush=True)
