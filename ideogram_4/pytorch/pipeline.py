# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4 — end-to-end text-to-image pipeline for the imagegen harness.

Ideogram 4 is a 9.3B single-stream conditional DiT. Unlike the other image-gen
pipelines here it runs **two** transformer branches — a conditional and an
unconditional one — and both are heavy, so both run **tensor-parallel across a
multi-chip mesh** (Megatron-1D over a ``("batch", "model")`` mesh, the same shard
spec the model-runner ``tensor_parallel-inference`` test validates). The Qwen3-VL
text encoder, the scheduler and (by default) the VAE stay on CPU.

A single chip is not an option here: the 34-layer DiT hangs on Blackhole inside
tt-metal's mesh command queue (``enqueue_write_shards``) at full scale. Sharding
the SwiGLU MLP 4-way cuts the per-chip weight write to 53.3% and clears it.

Rather than reimplement the denoising loop, ``generate()`` drives the upstream
``Ideogram4Pipeline`` on CPU and routes only the two transformer forwards through
TP-sharded, ``torch.compile(backend="tt")`` modules on the mesh. This keeps
numerics identical to upstream.

Per-generate timing is recorded into ``self._perf`` in the model-agnostic schema
the imagegen benchmark harness reads::

    _perf = {
        "components": {<name>: seconds, ...},   # scalar per-stage TT times
        "steps": [seconds, ...],                # per transformer-forward times
        "step_metric_name": "transformer_forward",
        "total": seconds,                       # full generate() wall time
    }

Note that classifier-free guidance means **two** transformer forwards per
denoising step, so ``len(_perf["steps"])`` is ``2 * num_inference_steps``.

Every ``ideogram4`` import is deliberately lazy (inside a function). The package
is installed per-model at test time, and a module-level import would break loader
discovery, which imports this package before those requirements exist.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from .src.model_utils import (
    DTYPE,
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    _load_sharded_state_dict,
    materialize_fp8_state_dict_to_bf16,
    shard_transformer_specs,
)

MODEL_ID = REPO_ID
# 512x512 is the validated shape for the tensor-parallel component test; the
# transformer's packed-sequence layout is derived from it in model_utils.
HEIGHT = 512
WIDTH = 512
GUIDANCE_SCALE = 7.0

# Optional on-disk cache for the materialized bf16 weights. The FP8 -> bf16
# conversion is single-threaded and takes ~30 minutes per branch, so a warm cache
# turns a 1-hour setup into seconds. Off unless a directory is given.
BF16_CACHE_ENV = "IDEOGRAM4_BF16_CACHE"


def _enable_spmd() -> None:
    """Enable torch_xla SPMD (shardy) — required before any device op.

    Mirrors ``tests/infra/utilities/torch_multichip_utils.enable_spmd`` but is
    inlined so this module carries no tt-xla test dependency.
    """
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def patch_mrope() -> None:
    """Rewrite ``Ideogram4MRoPE.forward``'s in-place advanced-index scatter.

    TEMPORARY. The upstream form does ``freqs_t[..., idx] = freqs[axis][..., idx]``,
    which mislowers on TT: it produces a position-correlated error in cos/sin.
    Because rope is recomputed every denoising step from the same ``position_ids``,
    that error resonates instead of averaging out and the decoded image comes back
    as a pure grid at the 16px patch period (patch_size 2 x AE scale 8) with no
    content. Single-forward PCC stays at 0.9957, so a tensor-level check does not
    catch it — the FFT magnitude at the patch period does: CPU 61.4, TT before
    1890.0, TT after 68.7.

    The masked-sum form below is bit-exact to the original on CPU (verified) and
    lowers cleanly. It is applied unconditionally so a CPU golden run and a TT run
    stay numerically identical.

    This belongs upstream — either in the ``ideogram4`` package or as a lowering
    fix for the in-place advanced-index scatter in tt-mlir, which would help every
    model hitting the same pattern. It lives here so the pipeline produces correct
    output today; remove it once either lands.
    """
    from ideogram4.modeling_ideogram4 import Ideogram4MRoPE

    def forward(self, position_ids):
        batch_size, seq_len, _ = position_ids.shape
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
            3, batch_size, -1, 1
        )
        freqs = (inv_freq @ pos.unsqueeze(2)).transpose(2, 3)  # (3, B, L, hd/2)
        dim = freqs.shape[-1]
        sel = torch.zeros(dim, dtype=torch.long, device=freqs.device)
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            sel[torch.arange(offset, length, 3, device=freqs.device)] = axis
        freqs_t = (
            freqs[0] * (sel == 0).to(freqs.dtype)
            + freqs[1] * (sel == 1).to(freqs.dtype)
            + freqs[2] * (sel == 2).to(freqs.dtype)
        )
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()

    Ideogram4MRoPE.forward = forward
    logger.info("[Ideogram4] patched Ideogram4MRoPE.forward (masked-sum interleave)")


def _build_bf16_transformer(index_filename: str, cache_name: str, dtype: torch.dtype):
    """Build one DiT branch from the FP8 checkpoint, materialized to bf16."""
    from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

    model = Ideogram4Transformer(Ideogram4Config())

    cache_dir = os.environ.get(BF16_CACHE_ENV)
    cache_path = os.path.join(cache_dir, f"{cache_name}.pt") if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        logger.info(f"[Ideogram4] loading cached bf16 {cache_name}")
        state_dict = torch.load(cache_path, map_location="cpu")
    else:
        logger.info(
            f"[Ideogram4] materializing fp8->bf16 for {cache_name} "
            "(single-threaded, ~30 min)"
        )
        raw = _load_sharded_state_dict(REPO_ID, index_filename)
        state_dict = materialize_fp8_state_dict_to_bf16(raw)
        del raw
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(state_dict, cache_path)

    model.load_state_dict(state_dict, strict=True)
    del state_dict
    return model.to(dtype).eval()


class Ideogram4Config:
    """Configuration for the Ideogram 4 text-to-image pipeline."""

    def __init__(
        self,
        dtype: torch.dtype = DTYPE,
        vae_on_tt: bool = False,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = MODEL_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.dtype = dtype
        # The VAE decoder is 49.6M params and fits unsharded on one chip. It is
        # a validated standalone component, but the default here keeps it on CPU
        # so the pipeline isolates the DiT — matching the SD1.5 / SD3 default.
        self.vae_on_tt = vae_on_tt
        # Harness-set XLA compile options (applied globally by the benchmark
        # harness via torch_xla.set_custom_compile_options before build).
        self.compile_options = compile_options or {}


class Ideogram4Pipeline:
    """Ideogram 4: both DiT branches tensor-parallel on the mesh, encoder on CPU."""

    def __init__(self, config: Ideogram4Config):
        self.config = config
        self.pipe = None
        self.mesh = None
        self._perf = None
        self._device = None

    def setup(self):
        """Build the mesh, load Ideogram 4, shard + compile both DiT branches."""
        from huggingface_hub import hf_hub_download
        from ideogram4.pipeline_ideogram4 import (
            Ideogram4Pipeline as UpstreamPipeline,
        )
        from ideogram4.pipeline_ideogram4 import (
            Ideogram4PipelineConfig,
            _load_autoencoder,
            _load_qwen3_vl,
        )

        patch_mrope()
        _enable_spmd()
        num_devices = xr.global_runtime_device_count()

        dtype = self.config.dtype
        upstream_config = Ideogram4PipelineConfig(weights_repo=REPO_ID)
        cpu = torch.device("cpu")

        conditional = _build_bf16_transformer(
            upstream_config.conditional_index_filename, "conditional", dtype
        )
        unconditional = _build_bf16_transformer(
            upstream_config.unconditional_index_filename, "unconditional", dtype
        )

        logger.info("[Ideogram4] loading Qwen3-VL text encoder (CPU)")
        tokenizer, text_encoder = _load_qwen3_vl(
            REPO_ID,
            cpu,
            dtype,
            tokenizer_subfolder=upstream_config.tokenizer_subfolder,
            text_encoder_subfolder=upstream_config.text_encoder_subfolder,
        )
        autoencoder = _load_autoencoder(
            hf_hub_download(
                repo_id=REPO_ID, filename=upstream_config.autoencoder_filename
            ),
            cpu,
            dtype,
        )

        self.pipe = UpstreamPipeline(
            conditional_transformer=conditional,
            unconditional_transformer=unconditional,
            text_encoder=text_encoder,
            text_tokenizer=tokenizer,
            autoencoder=autoencoder,
            config=upstream_config,
            device=cpu,
            dtype=dtype,
        )

        mesh_shape = _mesh_shape_for(num_devices)
        self.mesh = Mesh(np.array(range(num_devices)), mesh_shape, MESH_NAMES)
        xs.set_global_mesh(self.mesh)
        logger.info(
            f"[Ideogram4] mesh {mesh_shape} {MESH_NAMES} over {num_devices} chips"
        )

        self._device = torch_xla.device(0)
        self.pipe.conditional_transformer = self._to_device_sharded(
            conditional, "conditional"
        )
        self.pipe.unconditional_transformer = self._to_device_sharded(
            unconditional, "unconditional"
        )

        if self.config.vae_on_tt:
            self.pipe.autoencoder.decoder = self._vae_decoder_on_device(
                self.pipe.autoencoder.decoder
            )

    def _to_device_sharded(self, transformer, tag: str):
        """Move one DiT branch to the mesh, mark its shards, and compile it."""
        device = self._device
        transformer = transformer.to(device)

        # Build the shard spec AFTER moving to device so the dict keys are the
        # on-device parameters, then mark each. Attention cannot be head-parallel
        # here (18 heads do not divide the mesh, and qkv is fused), so only the
        # SwiGLU MLP shards; everything else stays replicated.
        specs = shard_transformer_specs(transformer)
        for param, spec in specs.items():
            xs.mark_sharding(param, self.mesh, spec)
        logger.info(f"[Ideogram4] {tag}: sharded {len(specs)} params (Megatron-1D)")

        compiled = torch.compile(transformer, backend="tt")

        class _Routed(torch.nn.Module):
            """Keeps the upstream pipeline's CPU-tensor calling convention."""

            def __init__(inner_self):
                super().__init__()
                inner_self.config = transformer.config

            def forward(inner_self, **kwargs):
                kwargs = {k: v.to(device) for k, v in kwargs.items()}
                start = time.perf_counter()
                out = compiled(**kwargs)
                # The .to("cpu") cast forces a device sync, so the timer captures
                # real device work for this forward.
                if isinstance(out, (list, tuple)):
                    out = type(out)([out[0].to("cpu"), *out[1:]])
                else:
                    out = out.to("cpu")
                if self._perf is not None:
                    self._perf["steps"].append(time.perf_counter() - start)
                return out

        return _Routed()

    def _vae_decoder_on_device(self, decoder):
        """Compile the VAE decoder on one chip (49.6M params, unsharded)."""
        device = self._device
        compiled = torch.compile(decoder.to(device), backend="tt")
        logger.info("[Ideogram4] VAE decoder on device (unsharded)")

        class _RoutedDecoder(torch.nn.Module):
            def forward(inner_self, latents):
                start = time.perf_counter()
                out = compiled(latents.to(device)).to("cpu")
                if self._perf is not None:
                    self._perf["components"]["vae_decode"] = time.perf_counter() - start
                return out

        return _RoutedDecoder()

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 28,
        seed: Optional[int] = 42,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> torch.Tensor:
        """Generate one image. Returns a ``(1, 3, H, W)`` float tensor in [0, 1]."""
        assert self.pipe is not None, "Call setup() before generate()."
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_forward",
            "total": None,
        }

        t_total = time.perf_counter()
        with torch.no_grad():
            images = self.pipe(
                prompt,
                height=self.config.height,
                width=self.config.width,
                num_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed if seed is not None else 0,
                raise_on_caption_issues=False,
            )
        self._perf["total"] = time.perf_counter() - t_total

        return _as_image_tensor(images[0])


def _mesh_shape_for(num_devices: int):
    """Return the ``(batch, model)`` mesh shape for a device count.

    Shares ``MESH_SHAPES`` with the component loader so the pipeline and the
    ``tensor_parallel-inference`` runner test cannot drift apart.
    """
    if num_devices not in MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count: {num_devices}. "
            f"Expected one of {sorted(MESH_SHAPES)}."
        )
    return MESH_SHAPES[num_devices]


def _as_image_tensor(image) -> torch.Tensor:
    """Normalize the upstream output (PIL image or tensor) to (1, 3, H, W) in [0, 1]."""
    if torch.is_tensor(image):
        tensor = image.detach().float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.clamp(0, 1)

    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
