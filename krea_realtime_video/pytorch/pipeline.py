# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Krea Realtime Video (CausalWanModel) e2e pipeline on Tenstorrent.

The transformer runs on TT (bf16), tensor-parallel sharded across the device
mesh and compiled with ``torch.compile(backend="tt")``; the text-encoder runs
once on TT then is freed; the VAE runs on CPU.

This pipeline can be reused by demo / benchmark / test.
"""

import importlib
import os
from collections import deque

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer

from .src.model_utils import (
    DTYPE,
    FRAME_SEQ_LENGTH,
    KREA_REPO_ID,
    KV_CACHE_NUM_FRAMES,
    LOCAL_ATTN_SIZE,
    MAX_SEQ_LEN,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    SEQ_LENGTH,
    WAN_REPO_ID,
    fixed_sinusoidal_embedding_1d,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_transformer_specs,
)
from .src.modified_model import apply_krea_static_patches

PROMPT = "a cat sitting on a boat"
NUM_INFERENCE_STEPS = 6
SEED = 42
HEIGHT = 480
WIDTH = 832
VAE_SCALE_FACTOR = 8
SHIFT = 5.0


def _tt(x):
    return x.to(device=xm.xla_device())


def _cpu(x):
    return x.to("cpu")


def init_kv_cache(num_blocks, num_heads, head_dim):
    """Per-block self-attention KV cache (also used by a test to build its twin)."""
    shape = [1, LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH, num_heads, head_dim]
    return [
        {
            "k": torch.zeros(shape, dtype=DTYPE).contiguous(),
            "v": torch.zeros(shape, dtype=DTYPE).contiguous(),
            "global_end_index": 0,
            "local_end_index": 0,
        }
        for _ in range(num_blocks)
    ]


def init_crossattn_cache(num_blocks, num_heads, head_dim):
    """Per-block cross-attention K/V cache (also used by a test to build its twin)."""
    shape = [1, MAX_SEQ_LEN, num_heads, head_dim]
    return [
        {
            "k": torch.zeros(shape, dtype=DTYPE),
            "v": torch.zeros(shape, dtype=DTYPE),
            "is_init": False,
        }
        for _ in range(num_blocks)
    ]


class KreaRealtimePipeline:
    """Krea e2e pipeline: transformer on TT, text-encoder (once) on TT, VAE on CPU."""

    def __init__(self, on_forward=None):
        # on_forward(kind, label, inputs, tt_out): optional per-forward validation
        # hook (default no-op). ``kind`` in {"encoder", "transformer", ...}; a test
        # runs a CPU reference on ``inputs`` and compares against ``tt_out``.
        self._on_forward = on_forward or (lambda *a, **k: None)

    def setup(self):
        self.text_encoder = load_text_encoder(WAN_REPO_ID, DTYPE)
        self.transformer = load_transformer(KREA_REPO_ID, DTYPE)
        self.vae = load_vae(WAN_REPO_ID, DTYPE)

        # Mandatory: upstream sinusoidal_embedding_1d hardcodes torch.cuda.current_device()
        # -> crashes on CPU/TT. Rebind the CPU-safe version on both the defining module
        # and the importer.
        _base = type(self.transformer).__module__
        for _mod in (_base, _base.rsplit(".", 1)[0] + ".model"):
            importlib.import_module(
                _mod
            ).sinusoidal_embedding_1d = fixed_sinusoidal_embedding_1d

        for blk in self.transformer.blocks:
            blk.self_attn.local_attn_size = -1
            blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK

        # Pin the KV-cache counters: they flip 0->4680 between denoising steps, so
        # torch.compile recompiles the self-attention on the 2nd forward (-> DRAM OOM).
        apply_krea_static_patches(self.transformer)

        self.tokenizer = AutoTokenizer.from_pretrained(
            WAN_REPO_ID, subfolder="tokenizer"
        )
        self.video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
        enc_mod = importlib.import_module(
            type(self.transformer).__module__.rsplit(".", 1)[0] + ".encoders"
        )
        self._prompt_clean = enc_mod.prompt_clean

        self._num_heads = self.transformer.config.num_heads
        self._head_dim = self.transformer.config.dim // self._num_heads
        self._num_tf_blocks = len(self.transformer.blocks)

        # SPMD mesh for the sharded transformer (inlined torch_xla; no main-module
        # dependency). Transformer sharded across the mesh; encoder + vae on 1 chip.
        xr.set_device_type("TT")
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(f"Unsupported device count {num_devices}")
        self.mesh_shape = MESH_SHAPES[num_devices]
        self._mesh = Mesh(np.array(range(num_devices)), self.mesh_shape, MESH_NAMES)
        self.transformer.compile(backend="tt")

    @staticmethod
    def _caches_to(caches, mover):
        for e in caches:
            e["k"] = mover(e["k"])
            e["v"] = mover(e["v"])

    # ────────────── text encoder (runs once on TT, then freed) ───────────────

    def _postprocess_embeds(self, embeds, seq_lens):
        embeds = embeds.to(dtype=DTYPE)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        return torch.stack(
            [
                torch.cat([u, u.new_zeros(MAX_SEQ_LEN - u.size(0), u.size(1))])
                for u in embeds
            ],
            dim=0,
        ).contiguous()

    def _encode(self, prompt):
        text_inputs = self.tokenizer(
            [self._prompt_clean(prompt)],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # Compile a LOCAL wrapper and drop it after use so its compiled device
        # buffers are released (in-place .compile() keeps them resident even after
        # .to("cpu"), which would OOM the transformer). The encoder runs once.
        self.text_encoder = self.text_encoder.to(xm.xla_device())
        compiled = torch.compile(self.text_encoder, backend="tt")
        tt_hidden = _cpu(compiled(_tt(input_ids), _tt(mask)).last_hidden_state)
        del compiled
        self.text_encoder = None
        torch_xla.sync()  # reclaim the encoder before the transformer lands

        self._on_forward(
            "encoder", "encoder", {"input_ids": input_ids, "mask": mask}, tt_hidden
        )
        return self._postprocess_embeds(tt_hidden, seq_lens)

    # ─────────────────── one transformer forward (on TT) ───────────────────

    def _transformer_step(
        self, label, x, t, context, kv_cache, crossattn_cache, current_start
    ):
        noise_tt = _cpu(
            self.transformer(
                x=_tt(x),
                t=_tt(t),
                context=_tt(context),
                kv_cache=kv_cache,
                seq_len=SEQ_LENGTH,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=None,
            )
        )
        self._on_forward(
            "transformer",
            label,
            {"x": x, "t": t, "context": context, "current_start": current_start},
            noise_tt,
        )
        return noise_tt

    # ──── block boundary (block >= 1): VAE-encode context + refill caches ────

    def _vae_encode(self, label, frames):
        # VAE runs on CPU -> no TT output to compare; the hook call is informational only.
        self.vae._enc_feat_map = [None] * 55
        lat = self.vae.encode(frames.to(self.vae.dtype)).latent_dist.mode()
        self._on_forward("vae_encode", label, {"frames": frames}, lat)
        z_dim = self.vae.config.z_dim
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(lat.device, lat.dtype)
        )
        std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, z_dim, 1, 1, 1
        ).to(lat.device, lat.dtype)
        return ((lat - mean) * std).to(DTYPE)

    def _build_context_frames(
        self, block_idx, current_denoised, frame_cache_context, block_latents
    ):
        total = (block_idx - 1) * NUM_FRAMES_PER_BLOCK
        if total < KV_CACHE_NUM_FRAMES:
            return current_denoised[:, :, :KV_CACHE_NUM_FRAMES]
        ctx = current_denoised[:, :, 1:][:, :, -KV_CACHE_NUM_FRAMES + 1 :]
        first = self._vae_encode(
            f"b{block_idx}_vae_encode", frame_cache_context[0].half()
        )
        first = first.to(block_latents)
        return torch.cat((first, ctx), dim=2)

    def _recompute(
        self, block_idx, context_frames, prompt_embeds, kv_cache, crossattn_cache
    ):
        ctx_ts = torch.zeros(
            (context_frames.shape[0], context_frames.shape[2]), dtype=torch.int64
        )
        self.transformer.block_mask = (
            self.transformer._prepare_blockwise_causal_attn_mask(
                xm.xla_device(),
                num_frames=context_frames.shape[2],
                frame_seqlen=FRAME_SEQ_LENGTH,
                num_frame_per_block=NUM_FRAMES_PER_BLOCK,
                local_attn_size=-1,
            )
        )
        noise_tt = _cpu(
            self.transformer(
                x=_tt(context_frames),
                t=_tt(ctx_ts),
                context=_tt(prompt_embeds),
                kv_cache=kv_cache,
                seq_len=SEQ_LENGTH,
                crossattn_cache=crossattn_cache,
                current_start=0,
                cache_start=None,
            )
        )
        self.transformer.block_mask = None
        self._on_forward(
            "recompute",
            f"b{block_idx}_recompute",
            {
                "context_frames": context_frames,
                "ctx_ts": ctx_ts,
                "prompt_embeds": prompt_embeds,
            },
            noise_tt,
        )

    # ─────────────────────────── VAE decode (CPU) ────────────────────────────

    def _decode(self, latents, block_idx, decoder_cache, frame_cache_context):
        if frame_cache_context is None:
            frame_cache_context = deque(maxlen=1 + (KV_CACHE_NUM_FRAMES - 1) * 4)

        z_dim = self.vae.config.z_dim
        mean = torch.tensor(self.vae.config.latents_mean, dtype=latents.dtype).view(
            1, z_dim, 1, 1, 1
        )
        std = 1.0 / torch.tensor(self.vae.config.latents_std, dtype=latents.dtype).view(
            1, z_dim, 1, 1, 1
        )
        rescaled = (latents / std + mean).to(self.vae.dtype)

        if block_idx == 0:
            self.vae.clear_cache()
            self.vae.clear_cache = lambda: None
            self.vae._feat_map = [None] * 55
        else:
            self.vae._feat_map = decoder_cache
        videos = self.vae.decode(rescaled, return_dict=False)[0]
        decoder_cache = self.vae._feat_map

        frame_cache_context.extend(videos.split(1, dim=2))
        frames = self.video_processor.postprocess_video(videos, output_type="pil")
        return frames[0], decoder_cache, frame_cache_context

    # ────────────────────── timesteps / latents / noise ──────────────────────

    def _set_timesteps(self, n):
        sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigmas = SHIFT * sigmas / (1 + (SHIFT - 1) * sigmas)
        timesteps = sigmas * 1000.0
        zero_padded = torch.cat([timesteps, torch.tensor([0])])
        denoising_steps = torch.linspace(1.0 * 1000, 0, n, dtype=torch.float32).to(
            torch.long
        )
        return zero_padded[1000 - denoising_steps], timesteps, sigmas

    def _prepare_init_latents(self, num_blocks, generator):
        shape = (
            1,
            NUM_CHANNELS_LATENTS,
            num_blocks * NUM_FRAMES_PER_BLOCK,
            HEIGHT // VAE_SCALE_FACTOR,
            WIDTH // VAE_SCALE_FACTOR,
        )
        return randn_tensor(
            shape, generator=generator, device="cpu", dtype=DTYPE
        ).contiguous()

    @staticmethod
    def _zero_kv_cache(kv_cache):
        for e in kv_cache:
            e["k"].zero_()
            e["v"].zero_()
            e["global_end_index"] = 0
            e["local_end_index"] = 0

    @staticmethod
    def _add_noise(sample, noise, timestep, all_timesteps, sigmas):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        tid = torch.argmin(
            (all_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[tid].reshape(-1, 1, 1, 1)
        return (
            (1 - sigma.double()) * sample.double() + sigma.double() * noise.double()
        ).type_as(noise)

    # ─────────────────────────────── generate ────────────────────────────────

    def generate(self, prompt, num_blocks, num_inference_steps, seed):
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(seed)

            prompt_embeds = self._encode(prompt)

            timesteps, all_timesteps, sigmas = self._set_timesteps(num_inference_steps)
            init_latents = self._prepare_init_latents(num_blocks, generator)

            kv_cache = init_kv_cache(
                self._num_tf_blocks, self._num_heads, self._head_dim
            )
            crossattn_cache = init_crossattn_cache(
                self._num_tf_blocks, self._num_heads, self._head_dim
            )
            decoder_cache = None
            frame_cache_context = None
            current_denoised = None

            # Transformer resident on the mesh for the whole run.
            self.transformer = self.transformer.to(xm.xla_device())
            for tensor, spec in shard_transformer_specs(self.transformer).items():
                xs.mark_sharding(tensor, self._mesh, spec)
            self._caches_to(kv_cache, _tt)
            self._caches_to(crossattn_cache, _tt)
            # Shard the caches on the head dim (dim 2) to match the transformer's
            # tensor-parallel head split; otherwise .to(xla) leaves them replicated
            # full-size on every chip -> DRAM OOM.
            head_spec = (None, None, "model", None)
            for e in (*kv_cache, *crossattn_cache):
                xs.mark_sharding(e["k"], self._mesh, head_spec)
                xs.mark_sharding(e["v"], self._mesh, head_spec)

            frames = []
            for block_idx in range(num_blocks):
                self._on_forward(
                    "block_start", f"b{block_idx}", {"block_idx": block_idx}, None
                )
                if block_idx > 0:
                    self._zero_kv_cache(kv_cache)

                start = block_idx * NUM_FRAMES_PER_BLOCK
                block_latents = init_latents[:, :, start : start + NUM_FRAMES_PER_BLOCK]
                current_start_frame = start

                if block_idx > 0:
                    context_frames = self._build_context_frames(
                        block_idx, current_denoised, frame_cache_context, block_latents
                    )
                    self._recompute(
                        block_idx,
                        context_frames,
                        prompt_embeds,
                        kv_cache,
                        crossattn_cache,
                    )

                latents = block_latents
                for i, t in enumerate(timesteps):
                    start_frame = min(current_start_frame, KV_CACHE_NUM_FRAMES)
                    noise = self._transformer_step(
                        f"b{block_idx}_step{i}",
                        latents,
                        t.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                        prompt_embeds,
                        kv_cache,
                        crossattn_cache,
                        start_frame * FRAME_SEQ_LENGTH,
                    )
                    tid = torch.argmin((all_timesteps - t).abs())
                    latents = (
                        latents.double() - sigmas[tid].double() * noise.double()
                    ).to(latents.dtype)
                    if i < num_inference_steps - 1:
                        t1 = timesteps[i + 1]
                        sample = latents.transpose(1, 2).squeeze(0)
                        noise_r = randn_tensor(
                            sample.shape,
                            device="cpu",
                            dtype=latents.dtype,
                            generator=generator,
                        )
                        latents = (
                            self._add_noise(
                                sample,
                                noise_r,
                                t1.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                                all_timesteps,
                                sigmas,
                            )
                            .unsqueeze(0)
                            .transpose(1, 2)
                        )
                current_denoised = latents

                block_frames, decoder_cache, frame_cache_context = self._decode(
                    current_denoised, block_idx, decoder_cache, frame_cache_context
                )
                frames.extend(block_frames)
            return frames
