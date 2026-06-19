# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LongCat-Image composite end-to-end generation on Tenstorrent device.

Wires the three brought-up components (Qwen2.5-VL text encoder, the MMDiT
denoiser, AutoencoderKL decoder) into the source LongCatImagePipeline and runs a
real prompt through them at the NATIVE 1024x1024 resolution.

Design (matches the model-bringup composite contract):
  * The scheduler / denoising loop, latent preparation, classifier-free-guidance
    math and cfg-renorm all stay in HOST PYTHON on CPU -- the pipeline's own
    loop, lifted here so the glue is never compiled as a device graph.
  * Only the three compiled components run on the TT device.
  * Each component runs in its OWN PROCESS (stage). A p150 chip has 32 GB DRAM;
    the text encoder (~15 GB) and transformer (~12 GB) do not co-reside, and
    torch_xla does not release device DRAM mid-process, so stages are isolated
    by process boundary. Intermediates are passed on disk.

Prompt rewrite is disabled: it is an autoregressive text_encoder.generate()
host-side convenience that does not change the output resolution.

Usage:  python composite_generate.py <stage> <workdir> [out_png]
        stage in {text_encoder, denoise, vae}
"""

import os
import sys
import time

import numpy as np
import torch

REPO_ID = "meituan-longcat/LongCat-Image"
STEPS = int(os.environ.get("LONGCAT_STEPS", "20"))
GUIDANCE = 4.5
HEIGHT = WIDTH = 1024  # native default (default_sample_size 128 * vae_scale_factor 8)
PROMPT = (
    "A photorealistic close-up of a fluffy orange cat wearing tiny round "
    "glasses, sitting at a wooden desk and reading a book, warm window light."
)

STAGE = sys.argv[1]
WORKDIR = sys.argv[2]
OUT_PATH = sys.argv[3] if len(sys.argv) > 3 else os.path.join(WORKDIR, "generated.png")
os.makedirs(WORKDIR, exist_ok=True)

# Inference only: disable autograd globally so torch_xla does not build gradient
# buffers (which would roughly double device activation memory and OOM the chip).
torch.set_grad_enabled(False)


def get_device():
    import torch_xla
    import torch_xla.runtime as xr

    xr.runtime.set_device_type("TT")
    if not torch_xla._XLAC._xla_computation_cache_is_initialized():
        xr.initialize_cache(f"{os.getcwd()}/tmp/")
    return torch_xla.device(0)


def sync():
    import torch_xla.core.xla_model as xm

    xm.mark_step()


# ---------------------------------------------------------------------------
if STAGE == "text_encoder":
    from diffusers import LongCatImagePipeline

    device = get_device()
    pipe = LongCatImagePipeline.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16)
    pipe.text_encoder = pipe.text_encoder.eval().to(device)
    pipe._guidance_scale = GUIDANCE
    t0 = time.time()
    prompt_embeds, text_ids = pipe.encode_prompt(prompt=PROMPT)
    neg_embeds, neg_text_ids = pipe.encode_prompt(prompt="")
    sync()
    torch.save(
        {
            "prompt_embeds": prompt_embeds.cpu(),
            "text_ids": text_ids.cpu(),
            "neg_embeds": neg_embeds.cpu(),
            "neg_text_ids": neg_text_ids.cpu(),
        },
        os.path.join(WORKDIR, "embeds.pt"),
    )
    print(f"[text_encoder] done in {time.time()-t0:.0f}s  embeds {tuple(prompt_embeds.shape)}", flush=True)

# ---------------------------------------------------------------------------
elif STAGE == "denoise":
    from diffusers.pipelines.longcat_image.pipeline_longcat_image import (
        LongCatImagePipeline,
        calculate_shift,
        retrieve_timesteps,
    )

    device = get_device()
    pipe = LongCatImagePipeline.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16)
    dtype = pipe.transformer.dtype
    e = torch.load(os.path.join(WORKDIR, "embeds.pt"))

    # latents + timesteps on CPU (host glue)
    latents, latent_image_ids = pipe.prepare_latents(
        1, 16, HEIGHT, WIDTH, dtype, torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )
    sigmas = np.linspace(1.0, 1.0 / STEPS, STEPS)
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_steps = retrieve_timesteps(
        pipe.scheduler, STEPS, torch.device("cpu"), sigmas=sigmas, mu=mu
    )
    print(f"[denoise] latents {tuple(latents.shape)}  steps {num_steps}", flush=True)

    pipe.transformer = pipe.transformer.eval().to(device)
    text_ids_d = e["text_ids"].to(device)
    neg_text_ids_d = e["neg_text_ids"].to(device)
    img_ids_d = latent_image_ids.to(device)
    pe_d = e["prompt_embeds"].to(device)
    ne_d = e["neg_embeds"].to(device)

    t_loop = time.time()
    for i, t in enumerate(timesteps):
        ts = (t.expand(latents.shape[0]).to(dtype) / 1000).to(device)
        hs = latents.to(device)
        nt = pipe.transformer(
            hidden_states=hs, timestep=ts, guidance=None,
            encoder_hidden_states=pe_d, txt_ids=text_ids_d, img_ids=img_ids_d,
            return_dict=False,
        )[0]
        sync()
        nu = pipe.transformer(
            hidden_states=hs, timestep=ts, guidance=None,
            encoder_hidden_states=ne_d, txt_ids=neg_text_ids_d, img_ids=img_ids_d,
            return_dict=False,
        )[0]
        sync()
        noise_text, noise_uncond = nt.cpu(), nu.cpu()
        noise_pred = noise_uncond + GUIDANCE * (noise_text - noise_uncond)
        cond_norm = torch.norm(noise_text, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=0.0, max=1.0)
        noise_pred = noise_pred * scale
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if i % 5 == 0:
            print(f"  step {i}/{num_steps}  ({time.time()-t_loop:.0f}s)", flush=True)

    torch.save({"latents": latents.cpu()}, os.path.join(WORKDIR, "latents.pt"))
    print(f"[denoise] {num_steps} steps done in {time.time()-t_loop:.0f}s", flush=True)

# ---------------------------------------------------------------------------
elif STAGE == "vae":
    from diffusers import LongCatImagePipeline

    device = get_device()
    pipe = LongCatImagePipeline.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16)
    dtype = pipe.vae.dtype
    latents = torch.load(os.path.join(WORKDIR, "latents.pt"))["latents"]

    latents = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    pipe.vae = pipe.vae.eval().to(device)
    t0 = time.time()
    image = pipe.vae.decode(latents.to(dtype).to(device), return_dict=False)[0]
    sync()
    image = image.cpu()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    image.save(OUT_PATH)
    print(f"[vae] decode {time.time()-t0:.0f}s  SAVED {OUT_PATH}  size={image.size}", flush=True)

else:
    raise SystemExit(f"unknown stage {STAGE!r}")
