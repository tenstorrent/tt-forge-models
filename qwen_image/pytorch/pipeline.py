# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image text-to-image pipeline running on Tenstorrent.

Every compute module runs on the TT backend, orchestrated by the diffusers
``QwenImagePipeline``: the Qwen2.5-VL text encoder and the ~20B QwenImage MMDiT
transformer are both tensor-parallel sharded across the mesh model axis, and the
VAE decoder is replicated. Only the tokenizer and the scheduler stay on host.

Each heavy module is placed and evicted in turn, so peak device DRAM is
``max(component)`` rather than their sum: the text encoder produces the prompt
embeddings and is moved back to host before the transformer is placed, and the
VAE is placed lazily on its first decode so it never coexists with the
transformer's denoising peak.

This is the reusable implementation that the runnable example
(``examples/pytorch/qwen_image.py``) consumes as a thin wrapper. Per-component
and per-step times go into ``self._perf`` after each ``generate()``.
"""

import gc
import time
from types import SimpleNamespace
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import QwenImagePipeline as DiffusersQwenImagePipeline
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image

from .loader import ModelLoader, ModelVariant
from .src.model_utils import (
    DTYPE,
    HEIGHT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    POSITIVE_MAGIC,
    PROMPT,
    REPO_ID,
    SEED,
    TOKENIZER_MAX_LENGTH,
    TRUE_CFG_SCALE,
    WIDTH,
    shard_text_encoder_specs,
    shard_transformer_specs,
)


class _DeviceTextEncoder:
    """Text encoder on TT (tensor-parallel sharded); returns hidden_states[-1].

    Stands in for ``QwenImagePipeline.text_encoder``, so it exposes the ``config``
    and ``dtype`` attributes the pipeline reads off the real module.
    """

    def __init__(self, text_encoder, mesh):
        self._dev = torch_xla.device()
        self.dtype = next(text_encoder.parameters()).dtype
        self.config = text_encoder.config
        text_encoder = text_encoder.to(self._dev)
        if hasattr(text_encoder, "tie_weights"):
            text_encoder.tie_weights()
        # Replicated the encoder is ~16.6 GB/chip and cannot coexist with the
        # transformer; sharding drops it to ~4 GB/chip.
        for tensor, spec in shard_text_encoder_specs(text_encoder).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(text_encoder, backend="tt")

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True):
        out = self._compiled(
            input_ids=input_ids.to(self._dev),
            attention_mask=(
                attention_mask.to(self._dev) if attention_mask is not None else None
            ),
            output_hidden_states=True,
        )
        # Only hidden_states[-1] is consumed downstream.
        return SimpleNamespace(hidden_states=(out.hidden_states[-1].cpu(),))


class _DeviceDenoiser:
    """Transformer on TT (tensor-parallel sharded); one call is one forward."""

    def __init__(self, transformer, mesh, forward_times):
        self._dev = torch_xla.device()
        self._forward_times = forward_times
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self.cache_context = transformer.cache_context

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        # return_dict=False -> 1-tuple; .cpu() is the sync point: it forces the
        # graph to run and only returns once the result is on host.
        (sample,) = self._compiled(**moved)
        sample = sample.cpu()
        self._forward_times.append(time.perf_counter() - t0)
        return (sample,)


class _DeviceVAEDecoder:
    """VAE decode on TT (replicated). Stashes the raw frame-0 pixels."""

    def __init__(self, vae, perf):
        self._dev = torch_xla.device()
        self._perf = perf
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.temperal_downsample = vae.temperal_downsample
        self._vae = vae
        self._compiled = None
        self.last_pixels = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep the VAE off-device during the denoise loop
        # so it never coexists with the transformer's peak.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        t0 = time.perf_counter()
        image = self._compiled(latents.to(self._dev)).cpu()
        self._perf["components"]["vae"] = time.perf_counter() - t0
        # The pipeline consumes decode(...)[0][:, :, 0]; stash that raw (B,3,H,W).
        self.last_pixels = image[:, :, 0]
        return (image,)


class QwenImageConfig:
    def __init__(
        self,
        height: int = HEIGHT,
        width: int = WIDTH,
        compile_options: Optional[dict] = None,
    ):
        self.repo_id = REPO_ID
        self.height = height
        self.width = width
        self.max_sequence_length = TOKENIZER_MAX_LENGTH
        # Applied globally by the caller; carried here for reference.
        self.compile_options = compile_options or {}


class QwenImagePipeline:
    """Text encoder + transformer (both sharded) and VAE (replicated) on TT.

    Built once with ``setup()``; ``generate()`` can be called repeatedly. The
    prompt embeddings and the placed modules are built on the first ``generate()``
    and reused afterwards: the stack does not free a compiled module's device
    memory, so re-placing per call would leave two full pipelines resident and
    OOM the VAE decode.
    """

    def __init__(self, config: QwenImageConfig):
        self.config = config
        # Persistent perf dict: the device wrappers hold this reference, so it is
        # cleared in place (never reassigned) between calls.
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        # Raw per-forward times; collapsed into per-step entries in generate().
        self._forward_times = []
        self._placed = False

    def setup(self):
        enable_spmd()
        self.num_devices = xr.global_runtime_device_count()
        self.mesh_shape, mesh_names = ModelLoader(
            ModelVariant.TRANSFORMER
        ).get_mesh_config(self.num_devices)
        self.mesh = get_mesh(self.mesh_shape, mesh_names)
        logger.info(
            "[setup] mesh {} over {} device(s)", self.mesh_shape, self.num_devices
        )

        self.pipe = DiffusersQwenImagePipeline.from_pretrained(
            self.config.repo_id, torch_dtype=DTYPE
        )
        self._raw_transformer = self.pipe.transformer
        self._raw_vae = self.pipe.vae

    def _encode(self, prompt: str):
        """Place the sharded text encoder, encode both prompts, then evict it."""
        logger.info("[STAGE] text_encoder (sharded): start")
        text_encoder = self.pipe.text_encoder
        self.pipe.text_encoder = _DeviceTextEncoder(text_encoder, self.mesh)

        # The masked-embedding extraction downstream of the encoder runs on host,
        # so encode against CPU tensors.
        cpu = torch.device("cpu")
        t0 = time.perf_counter()
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=prompt + POSITIVE_MAGIC,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
        )
        negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=NEGATIVE_PROMPT,
            device=cpu,
            num_images_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
        )
        elapsed = time.perf_counter() - t0

        # Evict the text encoder before the transformer is placed.
        self.pipe.text_encoder = text_encoder.to("cpu")
        gc.collect()
        torch_xla.sync()
        logger.info("[STAGE] text_encoder: done")

        return (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ), elapsed

    def generate(
        self,
        prompt: str = PROMPT,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        seed: Optional[int] = SEED,
    ) -> torch.Tensor:
        """End-to-end generation. Returns pixels in [-1, 1], shape (1, 3, H, W)."""
        self._perf["components"].clear()
        self._perf["steps"].clear()
        self._forward_times.clear()
        self._perf["total"] = None
        t_total_start = time.perf_counter()

        if not self._placed:
            self._embeds, encode_time = self._encode(prompt)
            self.pipe.transformer = _DeviceDenoiser(
                self._raw_transformer, self.mesh, self._forward_times
            )
            self._vae_wrapper = _DeviceVAEDecoder(self._raw_vae, self._perf)
            self.pipe.vae = self._vae_wrapper
            self._placed = True
            self._perf["components"]["text_encoder"] = encode_time

        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ) = self._embeds

        logger.info(
            "[STAGE] transformer (sharded) + vae: start ({} steps)",
            num_inference_steps,
        )
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            height=self.config.height,
            width=self.config.width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator,
        )
        logger.info("[STAGE] transformer + vae: done")

        per_step = 2 if TRUE_CFG_SCALE > 1.0 else 1
        self._perf["steps"].extend(
            sum(self._forward_times[i : i + per_step])
            for i in range(0, len(self._forward_times), per_step)
        )

        self._perf["total"] = time.perf_counter() - t_total_start
        return self._vae_wrapper.last_pixels


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale ([-1,1]→[0,255]), reshape and save the pipeline output as PNG."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)
